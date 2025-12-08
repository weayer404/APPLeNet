import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import setup_logger, set_random_seed, collect_env_info

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

def collect_features(trainer, max_samples=2000):
    model = trainer.model
    device = trainer.device
    model.eval()

    clip_feats = []
    actl_feats = []
    labels = []

    with torch.no_grad():
        for batch in trainer.dm.test_loader:
            img = batch["img"].to(device)
            lab = batch["label"].to(device)

            clip_f, actl_f = model.extract_features(img)

            clip_feats.append(clip_f.cpu().numpy())
            actl_feats.append(actl_f.cpu().numpy())
            labels.append(lab.cpu().numpy())

            # 防止样本太多，t-SNE 特别慢
            if sum(len(x) for x in labels) >= max_samples:
                break

    clip_feats = np.concatenate(clip_feats, axis=0)
    actl_feats = np.concatenate(actl_feats, axis=0)
    labels = np.concatenate(labels, axis=0)

    return clip_feats, actl_feats, labels


def plot_tsne(feats, labels, title, save_path):
    tsne = TSNE(n_components=2, init="pca", random_state=0, perplexity=30)
    feats_2d = tsne.fit_transform(feats)

    plt.figure(figsize=(8, 8))
    num_classes = len(np.unique(labels))
    for c in range(num_classes):
        idx = (labels == c)
        plt.scatter(
            feats_2d[idx, 0],
            feats_2d[idx, 1],
            s=5,
            alpha=0.7,
            label=str(c)
        )
    plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


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

    clip_feats, actl_feats, labels = collect_features(trainer, max_samples=2000)

    plot_tsne(
        clip_feats,
        labels,
        title="Original CLIP features",
        save_path=cfg.OUTPUT_DIR + "/tsne_clip.png"
    )
    plot_tsne(
        actl_feats,
        labels,
        title=cfg.TRAINER.NAME+" fused features",
        save_path=cfg.OUTPUT_DIR + "/tsne_" + cfg.TRAINER.NAME + ".png"
    )
    print("t-SNE 可视化已保存到:", cfg.OUTPUT_DIR)



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