import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import math
from collections import defaultdict

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from dassl.engine import build_trainer
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from train import setup_cfg

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except Exception as e:
    GradCAM = None
    show_cam_on_image = None
    ClassifierOutputTarget = None
    HAS_GRADCAM = False
    GRADCAM_IMPORT_ERROR = e


# PatternNet 默认用于“分类热力图/混淆矩阵热力图”的代表性类别。
# 这些类别覆盖线状结构、块状场景、目标型类别和自然背景，坐标轴更清晰，适合论文展示。
DEFAULT_CM_CLASSES = [
    "airplane",
    "runway",
    "bridge",
    "river",
    "harbor",
    "storage_tank",
    "parking_lot",
    "dense_residential",
    "forest",
    "freeway",
]

# Grad-CAM 默认也使用同一组代表性类别；如果觉得太多，可通过 --class-names 手动指定 5 个以上。
DEFAULT_GRADCAM_CLASSES = DEFAULT_CM_CLASSES

# CLIP image normalization
CLIP_MEAN = np.array([0.48145466, 0.45782750, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# ------------------------------------------------------------
# 中文显示与分类热力图样式配置
# 你后面想调字号，主要改 HEATMAP_STYLE 这一块即可。
# ------------------------------------------------------------
PATTERNNET_CN = {
    "airplane": "飞机",
    "baseballfield": "棒球场",
    "basketballcourt": "篮球场",
    "beach": "海滩",
    "bridge": "桥梁",
    "cemetery": "墓地",
    "chaparral": "灌丛地",
    "christmastreefarm": "圣诞树农场",
    "closedroad": "封闭道路",
    "coastalmansion": "海滨别墅",
    "crosswalk": "人行横道",
    "denseresidential": "密集住宅区",
    "ferryterminal": "渡轮码头",
    "footballfield": "足球场",
    "forest": "森林",
    "freeway": "高速公路",
    "golfcourse": "高尔夫球场",
    "harbor": "港口",
    "intersection": "交叉路口",
    "mobilehomepark": "移动房屋区",
    "nursinghome": "养老院",
    "oilgasfield": "油气田",
    "oilwell": "油井",
    "overpass": "立交桥",
    "parkinglot": "停车场",
    "parkingspace": "停车位",
    "railway": "铁路",
    "river": "河流",
    "runway": "跑道",
    "runwaymarking": "跑道标记",
    "shippingyard": "货运场",
    "solarpanel": "太阳能板",
    "sparseresidential": "稀疏住宅区",
    "storagetank": "储罐",
    "swimmingpool": "游泳池",
    "tenniscourt": "网球场",
    "transformerstation": "变电站",
    "wastewatertreatmentplant": "污水处理厂",
}

# 分类热力图中文字、图幅与字号设置。
# selected：默认 10 个代表类；all：--cm-all-classes 全 38 类。
HEATMAP_STYLE = {
    "selected": {
        "figsize": (13, 9),
        "dpi": 300,
        "title_fontsize": 20,
        "title_pad": 16,
        "xlabel": "预测标签",
        "ylabel": "真实标签",
        "axis_label_fontsize": 18,
        "axis_label_pad": 14,
        "tick_fontsize": 25,
        "xtick_rotation": 35,
        "ytick_rotation": 0,
        "xtick_ha": "right",
        "annotate": True,
        "annot_fontsize": 11,
        "colorbar_tick_fontsize": 25,
        "colorbar_label": "归一化比例",
        "colorbar_label_fontsize": 15,
        "cmap": "viridis",
        "tight_pad": 1.2,
    },
    "all": {
        "figsize": (24, 20),
        "dpi": 300,
        "title_fontsize": 22,
        "title_pad": 18,
        "xlabel": "预测标签",
        "ylabel": "真实标签",
        "axis_label_fontsize": 20,
        "axis_label_pad": 16,
        "tick_fontsize": 8,
        "xtick_rotation": 90,
        "ytick_rotation": 0,
        "xtick_ha": "center",
        "annotate": False,
        "annot_fontsize": 5,
        "colorbar_tick_fontsize": 14,
        "colorbar_label": "归一化比例",
        "colorbar_label_fontsize": 16,
        "cmap": "viridis",
        "tight_pad": 1.5,
    },
}


# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------
def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    for key in sorted(args.__dict__.keys()):
        print(f"{key}: {args.__dict__[key]}")
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_files(path):
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".csv")):
            os.remove(os.path.join(path, name))


def normalize_name(name: str) -> str:
    """Robust matching for PatternNet names: parking_lot / parking lot / Parking-Lot."""
    return str(name).lower().replace(" ", "").replace("_", "").replace("-", "")



def setup_chinese_font():
    """自动寻找可用中文字体，避免中文标题、坐标轴和类别名显示成方块。"""
    candidate_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidate_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print("[WARN] 未检测到常见中文字体，中文可能显示为方块。建议安装 fonts-noto-cjk 后删除 ~/.cache/matplotlib 再重试。")
    return "DejaVu Sans"


def class_name_cn(name: str) -> str:
    """把 PatternNet 英文类别名转成中文，兼容下划线、空格和短横线写法。"""
    key = normalize_name(name)
    return PATTERNNET_CN.get(key, str(name).replace("_", " "))


def classnames_to_cn(classnames):
    return [class_name_cn(name) for name in classnames]


def parse_csv_strs(text):
    if text is None or str(text).strip() == "":
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_csv_ints(text):
    if text is None or str(text).strip() == "":
        return []
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def safe_get_classnames(trainer):
    paths = [
        ("dm", "dataset", "classnames"),
        ("dm", "dataset", "class_names"),
        ("dm", "classnames"),
    ]
    for path in paths:
        obj = trainer
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and obj is not None:
            return list(obj)
    return None


def resolve_class_ids(
    classnames,
    class_ids_arg="",
    class_names_arg="",
    default_names=None,
    min_required=0,
    fill_to_default=True,
):
    """
    Priority: --class-ids > --class-names > default_names.
    For manual class selection, enforce min_required when needed.
    For default classes, if name variants do not match, fill with early dataset classes to avoid missing classes.
    """
    if classnames is None:
        raise RuntimeError("Cannot obtain class names from trainer.dm.dataset.classnames. Please check dataset class.")

    n_cls = len(classnames)
    name_to_id = {normalize_name(n): i for i, n in enumerate(classnames)}

    manual_ids = parse_csv_ints(class_ids_arg)
    manual_names = parse_csv_strs(class_names_arg)
    is_manual = bool(manual_ids or manual_names)

    if manual_ids:
        ids = []
        for cid in manual_ids:
            if 0 <= cid < n_cls and cid not in ids:
                ids.append(cid)
            else:
                print(f"[WARN] Ignore invalid or duplicated class id: {cid}")
        if len(ids) < min_required:
            raise ValueError(f"At least {min_required} classes are required, but only {len(ids)} valid class ids were provided.")
        return ids

    names = manual_names if manual_names else (default_names or [])
    ids = []
    missing = []
    for name in names:
        key = normalize_name(name)
        if key in name_to_id:
            cid = name_to_id[key]
            if cid not in ids:
                ids.append(cid)
        else:
            missing.append(name)

    if missing:
        print(f"[WARN] These class names were not found: {missing}")
        print("[INFO] Available PatternNet class names are:")
        print(", ".join(classnames))

    if is_manual and len(ids) < min_required:
        raise ValueError(f"At least {min_required} classes are required, but only {len(ids)} valid class names were matched.")

    if (not is_manual) and fill_to_default and default_names is not None and len(ids) < len(default_names):
        for cid in range(n_cls):
            if cid not in ids:
                ids.append(cid)
            if len(ids) >= len(default_names):
                break

    return ids


def inverse_clip_norm(img_tensor):
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    img = img * CLIP_STD + CLIP_MEAN
    return np.clip(img, 0.0, 1.0)


def sanitize_filename(text):
    text = str(text).replace(" ", "_").replace("/", "_").replace("\\", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch in ["_", "-", "."])


def write_selected_classes_csv(path, selected_ids, classnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["类别编号", "英文类别名", "中文类别名"])
        for cid in selected_ids:
            writer.writerow([cid, classnames[cid], class_name_cn(classnames[cid])])


# ------------------------------------------------------------
# Model forward utilities
# ------------------------------------------------------------
def get_base_model(trainer):
    model = trainer.model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model


def forward_logits(model, images, labels=None):
    """Compatible with model(image, label)->(logits, ...) and model(image)->logits."""
    try:
        if labels is not None:
            out = model(images, labels)
        else:
            out = model(images)
    except TypeError:
        out = model(images)

    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class CamWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        dummy_label = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        logits = forward_logits(self.model, x, dummy_label)
        return logits


def clip_vit_reshape_transform(tensor: torch.Tensor):
    """Grad-CAM reshape for CLIP ViT block outputs. Returns [B, C, H, W]."""
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim != 3:
        raise ValueError(f"Unexpected activation shape: {tuple(tensor.shape)}")

    if tensor.shape[0] > tensor.shape[1]:
        tensor = tensor.permute(1, 0, 2)  # [B, L, C]

    tensor = tensor[:, 1:, :]
    b, n, c = tensor.shape
    h = w = int(math.sqrt(n))
    if h * w != n:
        raise ValueError(f"Cannot reshape {n} tokens into square feature map.")
    return tensor.reshape(b, h, w, c).permute(0, 3, 1, 2)


def get_target_layer(base_model, layer_path):
    image_encoder = base_model.image_encoder
    return eval(f"image_encoder.{layer_path}")


# ------------------------------------------------------------
# Classification heatmap / confusion matrix
# ------------------------------------------------------------
def collect_predictions(trainer, base_model, device):
    loader = trainer.dm.test_loader
    y_true_all, y_pred_all, prob_all = [], [], []

    base_model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            logits = forward_logits(base_model, images, labels)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            y_true_all.extend(labels.detach().cpu().numpy().tolist())
            y_pred_all.extend(preds.detach().cpu().numpy().tolist())
            prob_all.extend(probs.max(dim=-1).values.detach().cpu().numpy().tolist())

    return np.array(y_true_all, dtype=np.int64), np.array(y_pred_all, dtype=np.int64), np.array(prob_all, dtype=np.float32)


def make_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def make_selected_confusion_matrix(y_true, y_pred, selected_ids, include_other=True):
    """
    Rows: selected true classes.
    Columns: selected predicted classes, plus optional Other column.
    If include_other=True, row normalization still reflects errors predicted outside selected classes.
    """
    selected_ids = list(selected_ids)
    row_map = {cid: i for i, cid in enumerate(selected_ids)}
    col_map = {cid: i for i, cid in enumerate(selected_ids)}
    n_rows = len(selected_ids)
    n_cols = len(selected_ids) + (1 if include_other else 0)
    other_col = n_cols - 1 if include_other else None

    cm = np.zeros((n_rows, n_cols), dtype=np.int64)
    mask = np.isin(y_true, selected_ids)
    selected_y_true = y_true[mask]
    selected_y_pred = y_pred[mask]

    for t, p in zip(selected_y_true, selected_y_pred):
        r = row_map[int(t)]
        if int(p) in col_map:
            c = col_map[int(p)]
            cm[r, c] += 1
        elif include_other:
            cm[r, other_col] += 1

    return cm, mask


def normalize_cm_rows(cm):
    denom = cm.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1)
    return cm.astype(np.float32) / denom.astype(np.float32)


def save_matrix_csv(path, matrix, row_labels, col_labels):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["真实\\预测"] + list(col_labels))
        for label, row in zip(row_labels, matrix):
            writer.writerow([label] + list(row))


def save_predictions_csv(path, y_true, y_pred, probs, classnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "真实类别编号", "真实类别名称", "预测类别编号", "预测类别名称", "预测概率", "是否正确"])
        for i, (t, p, prob) in enumerate(zip(y_true, y_pred, probs)):
            true_name = class_name_cn(classnames[int(t)]) if int(t) < len(classnames) else str(t)
            pred_name = class_name_cn(classnames[int(p)]) if int(p) < len(classnames) else str(p)
            writer.writerow([
                i,
                int(t), true_name,
                int(p), pred_name,
                float(prob), int(t == p),
            ])


def plot_heatmap(
    matrix_norm,
    row_labels,
    col_labels,
    save_path,
    title="PatternNet 分类热力图",
    dpi=300,
    annotate=True,
    axislabel_only=False,
):
    """
    绘制分类热力图。

    axislabel_only=False：常规论文版，保留图标题、x/y 轴标题、比例尺标题，并按需显示格内数值。
    axislabel_only=True ：多图对比版，只保留 x/y 轴刻度标签、比例尺数值刻度和图主体；
                          去掉图标题、x/y 轴标题、比例尺标题和格内数值。
    """
    setup_chinese_font()

    n_rows, n_cols = matrix_norm.shape
    style_name = "selected" if max(n_rows, n_cols) <= 15 else "all"
    style = HEATMAP_STYLE[style_name]
    fig_dpi = dpi if dpi is not None else style["dpi"]

    fig, ax = plt.subplots(figsize=style["figsize"], dpi=fig_dpi)
    im = ax.imshow(
        matrix_norm,
        interpolation="nearest",
        cmap=style["cmap"],
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    # 比例尺：两种版本都保留数值刻度；axislabel_only 版本只去掉“归一化比例”标题。
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=style["colorbar_tick_fontsize"])
    if not axislabel_only:
        cbar.set_label(
            style["colorbar_label"],
            fontsize=style["colorbar_label_fontsize"],
            labelpad=10,
        )
    else:
        cbar.set_label("")

    # x/y 轴刻度和类别标签：axislabel_only 版本仍然保留这些类别标签。
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(
        col_labels,
        rotation=style["xtick_rotation"],
        ha=style["xtick_ha"],
        fontsize=style["tick_fontsize"],
    )
    ax.set_yticklabels(
        row_labels,
        rotation=style["ytick_rotation"],
        fontsize=style["tick_fontsize"],
    )

    if not axislabel_only:
        ax.set_xlabel(style["xlabel"], fontsize=style["axis_label_fontsize"], labelpad=style["axis_label_pad"])
        ax.set_ylabel(style["ylabel"], fontsize=style["axis_label_fontsize"], labelpad=style["axis_label_pad"])
        ax.set_title(title, fontsize=style["title_fontsize"], pad=style["title_pad"])
    else:
        # 多图拼接版：去掉坐标轴名称和标题，只保留坐标轴类别刻度标签。
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    do_annotate = annotate and style["annotate"] and (not axislabel_only)
    if do_annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                v = float(matrix_norm[i, j])
                if v > 0:
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=style["annot_fontsize"],
                        color="white" if v > 0.45 else "black",
                    )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    fig.tight_layout(pad=style["tight_pad"])
    fig.savefig(save_path, dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig)

def generate_classification_heatmap(args, trainer, base_model, device, classnames):
    out_dir = os.path.join(args.output_dir, "classification_heatmap")
    ensure_dir(out_dir)
    if args.clean:
        clean_files(out_dir)

    print("[INFO] Collecting predictions for classification heatmap / confusion matrix ...")
    y_true, y_pred, probs = collect_predictions(trainer, base_model, device)
    n_cls = len(classnames)
    overall_acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
    print(f"[INFO] Test samples: {len(y_true)} | Num classes: {n_cls} | Overall ACC: {overall_acc * 100:.2f}%")

    save_predictions_csv(os.path.join(out_dir, "predictions.csv"), y_true, y_pred, probs, classnames)

    if args.cm_all_classes:
        full_cm = make_confusion_matrix(y_true, y_pred, n_cls)
        full_norm = normalize_cm_rows(full_cm)
        classnames_cn = classnames_to_cn(classnames)
        save_matrix_csv(os.path.join(out_dir, "confusion_matrix_all_count.csv"), full_cm, classnames_cn, classnames_cn)
        save_matrix_csv(os.path.join(out_dir, "confusion_matrix_all_norm.csv"), full_norm, classnames_cn, classnames_cn)

        title = args.cm_title if args.cm_title else f"PatternNet 全类别分类热力图（整体准确率={overall_acc * 100:.2f}%）"
        plot_heatmap(
            full_norm,
            classnames_cn,
            classnames_cn,
            os.path.join(out_dir, "confusion_matrix_all_names.png"),
            title=title,
            dpi=args.dpi,
            annotate=False,
        )
        plot_heatmap(
            full_norm,
            classnames_cn,
            classnames_cn,
            os.path.join(out_dir, "confusion_matrix_all_names_axislabel_only.png"),
            title=title,
            dpi=args.dpi,
            annotate=False,
            axislabel_only=True,
        )
        plot_heatmap(
            full_norm,
            [str(i) for i in range(n_cls)],
            [str(i) for i in range(n_cls)],
            os.path.join(out_dir, "confusion_matrix_all_ids.png"),
            title=title,
            dpi=args.dpi,
            annotate=False,
        )
        plot_heatmap(
            full_norm,
            [str(i) for i in range(n_cls)],
            [str(i) for i in range(n_cls)],
            os.path.join(out_dir, "confusion_matrix_all_ids_axislabel_only.png"),
            title=title,
            dpi=args.dpi,
            annotate=False,
            axislabel_only=True,
        )
        print(f"[INFO] Full-class classification heatmap saved to: {out_dir}")
        return

    # Default: selected representative classes only.
    selected_ids = resolve_class_ids(
        classnames,
        class_ids_arg=args.cm_class_ids,
        class_names_arg=args.cm_class_names,
        default_names=DEFAULT_CM_CLASSES,
        min_required=5,
        fill_to_default=True,
    )
    selected_names = [class_name_cn(classnames[cid]) for cid in selected_ids]
    write_selected_classes_csv(os.path.join(out_dir, "selected_classes.csv"), selected_ids, classnames)

    print("[INFO] Selected classes for classification heatmap:")
    for cid in selected_ids:
        print(f"  {cid}: {classnames[cid]}")

    cm_sel, selected_mask = make_selected_confusion_matrix(
        y_true, y_pred, selected_ids, include_other=not args.cm_no_other
    )
    cm_sel_norm = normalize_cm_rows(cm_sel)
    col_labels = selected_names + ([] if args.cm_no_other else ["其他"])
    row_labels = selected_names

    selected_acc = float((y_true[selected_mask] == y_pred[selected_mask]).mean()) if selected_mask.any() else 0.0
    print(f"[INFO] Selected-class samples: {int(selected_mask.sum())} | Selected ACC: {selected_acc * 100:.2f}%")

    save_matrix_csv(os.path.join(out_dir, "confusion_matrix_selected_count.csv"), cm_sel, row_labels, col_labels)
    save_matrix_csv(os.path.join(out_dir, "confusion_matrix_selected_norm.csv"), cm_sel_norm, row_labels, col_labels)

    title = args.cm_title if args.cm_title else (
        f"PatternNet 代表类别分类热力图"
        f"（整体准确率={overall_acc * 100:.2f}%，所选类别准确率={selected_acc * 100:.2f}%）"
    )
    plot_heatmap(
        cm_sel_norm,
        row_labels,
        col_labels,
        os.path.join(out_dir, "confusion_matrix_selected_names.png"),
        title=title,
        dpi=args.dpi,
        annotate=not args.no_annotate,
    )
    plot_heatmap(
        cm_sel_norm,
        row_labels,
        col_labels,
        os.path.join(out_dir, "confusion_matrix_selected_names_axislabel_only.png"),
        title=title,
        dpi=args.dpi,
        annotate=False,
        axislabel_only=True,
    )

    # Also save an ID-version for compact insertion if names still feel long.
    id_row_labels = [str(cid) for cid in selected_ids]
    id_col_labels = [str(cid) for cid in selected_ids] + ([] if args.cm_no_other else ["其他"])
    plot_heatmap(
        cm_sel_norm,
        id_row_labels,
        id_col_labels,
        os.path.join(out_dir, "confusion_matrix_selected_ids.png"),
        title=title,
        dpi=args.dpi,
        annotate=not args.no_annotate,
    )
    plot_heatmap(
        cm_sel_norm,
        id_row_labels,
        id_col_labels,
        os.path.join(out_dir, "confusion_matrix_selected_ids_axislabel_only.png"),
        title=title,
        dpi=args.dpi,
        annotate=False,
        axislabel_only=True,
    )

    print(f"[INFO] Selected-class classification heatmap saved to: {out_dir}")


# ------------------------------------------------------------
# Grad-CAM / attention visualization
# ------------------------------------------------------------
def generate_gradcam(args, trainer, base_model, device, classnames):
    if not HAS_GRADCAM:
        print(f"[WARN] pytorch_grad_cam import failed: {GRADCAM_IMPORT_ERROR}")
        print("[WARN] Skip Grad-CAM generation. Classification heatmap has already been generated.")
        return

    selected_class_ids = resolve_class_ids(
        classnames,
        class_ids_arg=args.class_ids,
        class_names_arg=args.class_names,
        default_names=DEFAULT_GRADCAM_CLASSES,
        min_required=5,
        fill_to_default=True,
    )

    if not selected_class_ids:
        print("[WARN] No class selected for Grad-CAM. Skip Grad-CAM.")
        return

    print("[INFO] Selected Grad-CAM classes:")
    for cid in selected_class_ids:
        print(f"  {cid}: {classnames[cid]}")

    out_dir = os.path.join(args.output_dir, "gradcam")
    original_dir = os.path.join(out_dir, "original")
    cam_dir = os.path.join(out_dir, "cam")
    pair_dir = os.path.join(out_dir, "pair")
    for d in [out_dir, original_dir, cam_dir, pair_dir]:
        ensure_dir(d)
        if args.clean:
            clean_files(d)

    target_layer = get_target_layer(base_model, args.target_layer)
    use_reshape = ("transformer" in args.target_layer) or ("resblocks" in args.target_layer) or ("ln_" in args.target_layer)

    cam_model = CamWrapper(base_model).to(device)
    cam_model.eval()
    cam = GradCAM(
        model=cam_model,
        target_layers=[target_layer],
        reshape_transform=clip_vit_reshape_transform if use_reshape else None,
    )

    selected_set = set(selected_class_ids)
    class_counts = defaultdict(int)
    total_saved = 0
    rows = []

    loader = trainer.dm.test_loader
    print("[INFO] Generating Grad-CAM images ...")
    for batch_idx, batch in enumerate(loader):
        images = batch["img"].to(device)
        labels = batch["label"].to(device)

        for i in range(images.size(0)):
            label = int(labels[i].item())
            if label not in selected_set:
                continue
            if class_counts[label] >= args.num_images:
                continue

            img_tensor = images[i:i+1].clone().requires_grad_(True)
            gt_name = classnames[label]

            with torch.enable_grad():
                logits = cam_model(img_tensor)
                probs = torch.softmax(logits, dim=-1)
                pred = int(torch.argmax(logits, dim=-1).item())
                pred_name = classnames[pred] if pred < len(classnames) else str(pred)
                pred_prob = float(probs[0, pred].item())

                target_id = label  # Use GT class as CAM target for fair comparison.
                grayscale_cam = cam(
                    input_tensor=img_tensor,
                    targets=[ClassifierOutputTarget(target_id)],
                )[0]

            rgb_img = inverse_clip_norm(img_tensor[0])
            overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            overlay_rgb = overlay.astype(np.float32) / 255.0

            base = f"class{label:02d}_{sanitize_filename(gt_name)}_k{class_counts[label]:02d}_pred_{sanitize_filename(pred_name)}"
            original_path = os.path.join(original_dir, base + "_original.png")
            cam_path = os.path.join(cam_dir, base + "_cam.png")
            pair_path = os.path.join(pair_dir, base + "_pair.png")

            cv2.imwrite(original_path, cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(cam_path, cv2.cvtColor((overlay_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            pair = np.concatenate([rgb_img, overlay_rgb], axis=1)
            cv2.imwrite(pair_path, cv2.cvtColor((pair * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            rows.append([total_saved, label, gt_name, pred, pred_name, pred_prob, original_path, cam_path, pair_path])
            class_counts[label] += 1
            total_saved += 1
            print(f"[INFO] Saved Grad-CAM: {pair_path}")

            if all(class_counts[cid] >= args.num_images for cid in selected_class_ids):
                break
        if all(class_counts[cid] >= args.num_images for cid in selected_class_ids):
            break

    index_path = os.path.join(out_dir, "index.csv")
    with open(index_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "gt_id", "gt_name", "pred_id", "pred_name", "pred_prob", "original_path", "cam_path", "pair_path"])
        writer.writerows(rows)

    for cid in selected_class_ids:
        if class_counts[cid] < args.num_images:
            print(f"[WARN] Class {cid} ({classnames[cid]}) only saved {class_counts[cid]} images; expected {args.num_images}.")

    print(f"[INFO] Grad-CAM saved to: {out_dir}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
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

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")
    trainer.model.to(device)
    base_model = get_base_model(trainer)
    base_model.to(device)
    base_model.eval()

    classnames = safe_get_classnames(trainer)
    if classnames is None:
        raise RuntimeError("Cannot obtain classnames from dataset. Please check PatternNet dataset class.")

    print(f"[INFO] Loaded {len(classnames)} classes:")
    print(", ".join(classnames))

    ensure_dir(args.output_dir)

    if not args.skip_cm:
        generate_classification_heatmap(args, trainer, base_model, device, classnames)

    if not args.only_cm:
        generate_gradcam(args, trainer, base_model, device, classnames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Keep your original train/test arguments for direct bash compatibility.
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="", help="path to trainer config file")
    parser.add_argument("--dataset-config-file", type=str, default="", help="path to dataset config file")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory")
    parser.add_argument("--load-epoch", type=int, default=None, help="load model weights at this epoch")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")

    # Grad-CAM arguments.
    parser.add_argument("--num-images", type=int, default=3, help="Grad-CAM images per selected class")
    parser.add_argument("--target-layer", type=str, default="transformer.resblocks[-1].ln_1", help="target layer under image_encoder")
    parser.add_argument("--class-names", type=str, default="", help="comma-separated PatternNet class names for Grad-CAM; at least 5 if specified")
    parser.add_argument("--class-ids", type=str, default="", help="comma-separated PatternNet class ids for Grad-CAM; at least 5 if specified")

    # Classification heatmap arguments.
    parser.add_argument("--only-cm", action="store_true", help="only generate classification heatmap / confusion matrix")
    parser.add_argument("--skip-cm", action="store_true", help="skip classification heatmap / confusion matrix")
    parser.add_argument("--cm-all-classes", action="store_true", help="draw the full 38-class PatternNet heatmap instead of the default representative subset")
    parser.add_argument("--cm-class-names", type=str, default="", help="comma-separated class names for classification heatmap; at least 5 if specified")
    parser.add_argument("--cm-class-ids", type=str, default="", help="comma-separated class ids for classification heatmap; at least 5 if specified")
    parser.add_argument("--cm-no-other", action="store_true", help="do not add the Other prediction column for selected-class heatmap")
    parser.add_argument("--cm-title", type=str, default="", help="title of classification heatmap")
    parser.add_argument("--no-annotate", action="store_true", help="do not print values inside cells for selected-class heatmap")
    parser.add_argument("--dpi", type=int, default=300, help="figure dpi")
    parser.add_argument("--clean", action="store_true", help="clean old png/csv files in output subfolders")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using command-line")
    args = parser.parse_args()
    main(args)
