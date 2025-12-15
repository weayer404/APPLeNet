import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import numpy as np
import cv2

from collections import defaultdict
from types import SimpleNamespace


from dassl.engine import build_trainer
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.engine import build_trainer
from train import setup_cfg

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def clip_vit_reshape_transform(tensor: torch.Tensor):
    """
    适配 CLIP ViT 的特征形状:
    - 对 ViT block 的输出: [L, B, C] -> [B, H, W, C]
    - 对 CNN/conv1 的输出: [B, C, H, W] -> [B, H, W, C]
    """
    if tensor.ndim == 3:
        # ViT: [L, B, C]
        # 去掉 class token
        tensor = tensor[1:, :, :]
        L, B, C = tensor.shape
        H = W = int(L ** 0.5)

        tensor = tensor.permute(1, 0, 2)       # [B, L, C]
        tensor = tensor.reshape(B, H, W, C)    # [B, H, W, C]
        return tensor
    elif tensor.ndim == 4:
        # CNN feature map: [B, C, H, W]
        return tensor.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Unexpected tensor dim: {tensor.shape}")

def print_args(args, cfg):
	print("***************")
	print("** Arguments **")
	print("***************")
	optkeys = list(args.__dict__.keys())
	optkeys.sort()
	for key in optkeys:
		print("{}: {}".format(key, args.__dict__[key]))
	print("************")
	print("** Config **")
	print("************")
	print(cfg)
     


class CamWrapper(torch.nn.Module):
    """
    将 CustomCLIP 打包成 logits-only 的模型，方便 Grad-CAM 调用。
    """
    def __init__(self, mpple_model: torch.nn.Module):
        super().__init__()
        self.model = mpple_model

    def forward(self, x):
        # dummy label，只是为了匹配 forward(image, label) 的接口
        dummy_label = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        logits, _, _ = self.model(x, dummy_label)
        return logits


def tensor_to_rgb(img_tensor: torch.Tensor):
    """
    把 dataloader 出来的 tensor(3,H,W) 转成 [0,1] 的 numpy RGB。
    """
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    return img

def clean_gradcam_folder(output_dir):
    """
    删除输出文件夹中之前生成的 Grad-CAM 图片
    仅删除符合命名规则的 .png 文件
    """
    if os.path.exists(output_dir):
        # 获取所有符合 gradcam_idxX_labelY.png 命名规则的 .png 文件
        files = [f for f in os.listdir(output_dir) if f.startswith('gradcam_idx') and f.endswith('.png')]
        for file in files:
            os.remove(os.path.join(output_dir, file))
            print(f"Deleted: {file}")
    else:
        print(f"Output directory {output_dir} does not exist.")



def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
        setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    trainer.model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(device)

    # 取内部的 CustomCLIP 模型（可能被 DataParallel 包了一层）
    base_model = trainer.model
    if isinstance(base_model, torch.nn.DataParallel):
        base_model = base_model.module

    image_encoder = base_model.image_encoder
   
    # 选择目标层：
    # 对 ViT-B/16 的 CLIP，一般可以用最后一个 transformer block 做 Grad-CAM
    # 默认 target_layer 传的是 "transformer.resblocks[-1].ln_1"
    if args.target_layer.strip():
        target_layer = eval(f"image_encoder.{args.target_layer}")
    else:
        # 如果你不传，就用一个比较通用的默认值（ViT 适用）
        target_layer = image_encoder.conv1

    # === 新增：临时解冻视觉编码器，让 Grad-CAM 能拿到梯度 ===
    # for p in base_model.image_encoder.parameters():
    #     p.requires_grad_(True)


    # Grad-CAM 用的包裹模型（只有 image -> logits）
    cam_model = CamWrapper(base_model).to(trainer.device)


    target_layers = [target_layer]

    # ViT 用 vit_reshape_transform，把 token 映射成 2D feature map
    cam = GradCAM(
        model=cam_model,
        target_layers=target_layers,
        reshape_transform=None    #clip_vit_reshape_transform
    )

    
    # 获取完整的测试集
    data_loader = trainer.dm.test_loader

    # 记录每个类别已可视化图片的数量
    class_counts = defaultdict(int)
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    clean_gradcam_folder(cfg.OUTPUT_DIR)

    # 只画每个类前 N 张图片，避免内存溢出
    num_per_class = args.num_images  # 每个类别需要展示的图像数

    # 逐批次处理，每个 batch 里处理并更新
    print(f"Generating Grad-CAM for {num_per_class} images per class ...")

    for batch in data_loader:
        images = batch["img"].to(device)
        images.requires_grad_(True)
        labels = batch["label"].to(device)

        for i in range(images.size(0)):
            img_tensor = images[i]
            label = int(labels[i].item())

            # 每个类只可视化前 N 张图片
            if class_counts[label] >= num_per_class:
                continue

            # 累加该类的已可视化图片数
            class_counts[label] += 1

            # 原图（0~1）
            rgb_img = tensor_to_rgb(img_tensor)

            # Grad-CAM 目标类别：默认用 GT label，你也可以改用预测类别
            targets = [ClassifierOutputTarget(label)]

            # cam() 会自动调用 cam_model.forward(x)
            grayscale_cam = cam(
                input_tensor=img_tensor.unsqueeze(0),
                targets=targets
            )[0]  # H x W, 0~1

            visualization = show_cam_on_image(
                rgb_img,
                grayscale_cam,
                use_rgb=True
            )

            # 保存成 BGR 格式给 cv2
            save_path = os.path.join(
                cfg.OUTPUT_DIR,
                f"gradcam_idx{i}_label{label}.png"
            )
            cv2.imwrite(
                save_path,
                cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            )
            print(f"Saved: {save_path}")



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--root", type=str, default="", help="path to dataset")
	parser.add_argument("--output-dir", type=str, default="", help="output directory")
	parser.add_argument(
		"--resume",
		type=str,
		default="",
		help="checkpoint directory (from which the training resumes)",
	)
	parser.add_argument(
		"--seed", type=int, default=-1, help="only positive value enables a fixed seed"
	)
	parser.add_argument(
		"--source-domains", type=str, nargs="+", help="source domains for DA/DG"
	)
	parser.add_argument(
		"--target-domains", type=str, nargs="+", help="target domains for DA/DG"
	)
	parser.add_argument(
		"--transforms", type=str, nargs="+", help="data augmentation methods"
	)
	parser.add_argument(
		"--config-file", type=str, default="", help="path to config file"
	)
	parser.add_argument(
		"--dataset-config-file",
		type=str,
		default="",
		help="path to config file for dataset setup",
	)
	parser.add_argument("--trainer", type=str, default="", help="name of trainer")
	parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
	parser.add_argument("--head", type=str, default="", help="name of head")
	parser.add_argument("--eval-only", action="store_true", help="evaluation only")
	parser.add_argument(
		"--model-dir",
		type=str,
		default="",
		help="load model from this directory for eval-only mode",
	)
	parser.add_argument(
		"--load-epoch", type=int, help="load model weights at this epoch for evaluation"
	)
	parser.add_argument(
		"--no-train", action="store_true", help="do not call trainer.train()"
	)
	parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="最多可视化多少张图片"
    )
	parser.add_argument(
        "--target-layer",
        type=str,
        default="transformer.resblocks[-1].ln_1",
        help=(
            "相对于 image_encoder 的层路径，比如：\n"
            "  ViT: 'transformer.resblocks[-1].ln_1'\n"
            "  ResNet: 'layer4[-1].conv3'\n"
        ),
	)
	parser.add_argument(
		"opts",
		default=None,
		nargs=argparse.REMAINDER,
		help="modify config options using the command-line",
	)
	args = parser.parse_args()
	main(args)