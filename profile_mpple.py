import argparse
import time
import torch

from train import setup_cfg
from dassl.engine import build_trainer
from thop import profile


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def benchmark_latency(model, device, input_size=(1, 3, 224, 224), warmup=50, repeat=200):
    model.eval()

    dummy_x = torch.randn(*input_size, device=device)
    dummy_y = torch.zeros(input_size[0], dtype=torch.long, device=device)

    # warmup
    for _ in range(warmup):
        _ = model(dummy_x, dummy_y)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        _ = model(dummy_x, dummy_y)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    ms_per_img = elapsed * 1000.0 / (repeat * input_size[0])
    img_per_sec = (repeat * input_size[0]) / elapsed

    return ms_per_img, img_per_sec


def compute_macs(model, device, input_size=(1, 3, 224, 224)):
    model.eval()

    dummy_x = torch.randn(*input_size, device=device)
    dummy_y = torch.zeros(input_size[0], dtype=torch.long, device=device)

    macs, params = profile(
        model,
        inputs=(dummy_x, dummy_y),
        verbose=False
    )
    return macs, params


def main(args):
    cfg = setup_cfg(args)
    trainer = build_trainer(cfg)

    if args.model_dir:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)

    model = trainer.model.to(trainer.device)
    device = trainer.device
    model.eval()

    h, w = cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1]
    input_size = (args.batch_size, 3, h, w)

    trainable_params = count_trainable_params(model)
    macs, total_params = compute_macs(model, device=device, input_size=input_size)
    ms_per_img, img_per_sec = benchmark_latency(
        model,
        device=device,
        input_size=input_size,
        warmup=args.warmup,
        repeat=args.repeat
    )

    print("=" * 60)
    print(f"Model: {cfg.TRAINER.NAME}")
    print(f"Input size: {input_size[0]} x 3 x {h} x {w}")
    print(f"Trainable Params: {trainable_params / 1e6:.3f} M")
    print(f"MACs: {macs / 1e9:.3f} G")
    print(f"Total Params (for reference): {total_params / 1e6:.3f} M")
    print(f"Pure forward latency: {ms_per_img:.3f} ms/img")
    print(f"Pure forward throughput: {img_per_sec:.3f} img/s")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--source-domains", type=str, nargs="+")
    parser.add_argument("--target-domains", type=str, nargs="+")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--transforms", type=str, nargs="+")
    parser.add_argument("--config-file", type=str, default="")
    parser.add_argument("--dataset-config-file", type=str, default="")
    parser.add_argument("--trainer", type=str, default="")
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--load-epoch", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    args = parser.parse_args()
    main(args)