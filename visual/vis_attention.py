import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dassl.engine import build_trainer
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.engine import build_trainer

from train import setup_cfg  # 如果你有类似函数，没有就照 train.py 抄一份配置构建代码


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
     
def get_attention_map(model, grid_size):
    """
    从 model.last_attention_weights 里取出对图像 token 的注意力，
    做简单的平均，reshape 成 (H, W) 的注意力图。
    """
    attn = model.last_attention_weights  # [B, T_text, T_img] 之类
    # 这里先简单平均：对 batch 取第 0 个，对文本 token 求平均
    attn_img = attn[0].mean(dim=0)  # -> [T_img]
    H, W = grid_size
    attn_img = attn_img.reshape(H, W)
    attn_img = attn_img - attn_img.min()
    attn_img = attn_img / (attn_img.max() + 1e-6)
    return attn_img.cpu().numpy()


def overlay_attention_on_image(img, attn_map, save_path):
    """
    img: numpy array, H x W x 3 (0-255)
    attn_map: h x w (0-1)
    """
    h, w = img.shape[:2]
    attn_resized = cv2.resize(attn_map, (w, h))
    heatmap = (attn_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = 0.5 * heatmap + 0.5 * img
    cv2.imwrite(save_path, overlay.astype(np.uint8))


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
    

    model = trainer.model
    model.eval()

    # 准备一张测试图片
    # 这里示意从 dataloader 里拿一张
    batch = next(iter(trainer.dm.test_loader))
    image = batch["img"].to(trainer.device)
    label = batch["label"].to(trainer.device)

    with torch.no_grad():
        _ = model(image, label)  # 前向一次，填充 last_attention_weights

    # 假设你的 visual backbone 是 7x7 patch
    attn_map = get_attention_map(model, grid_size=14)

    # 把 tensor 图片转回 numpy
    img_np = image[0].cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    save_path = os.path.join(cfg.OUTPUT_DIR, "attention_example.png")
    overlay_attention_on_image(img_np, attn_map, save_path)
    print("attention 可视化已保存到:", save_path)



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
		"opts",
		default=None,
		nargs=argparse.REMAINDER,
		help="modify config options using the command-line",
	)
	args = parser.parse_args()
	main(args)