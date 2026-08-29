#!/usr/bin/env python3
"""Render one GT-versus-prediction diagnostic image for each UA-DETRAC test sequence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DATA_ROOT = Path("/root/autodl-fs/datasets/UA-DETRAC_COCO")
DEFAULT_ANNOTATIONS = DATA_ROOT / "annotations/instances_test.json"
DEFAULT_PREDICTIONS = Path(
    "CaS-DETR/outputs/uadetrac/main/"
    "cas_detr_cass_moe_sg_ccff_cap2x_hgnetv2_s/predictions_test.json"
)
DEFAULT_OUTPUT = DEFAULT_PREDICTIONS.parent / "sequence_gt_prediction_comparisons"
SMALL_AREA = 32 * 32
PREDICTION_SCORE = 0.20
MAX_PREDICTIONS = 30
COLORS = {1: "#22c55e", 2: "#38bdf8", 3: "#f59e0b", 4: "#e879f9"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--images", type=Path, default=DATA_ROOT / "test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def iou(first: dict, second: dict) -> float:
    ax, ay, aw, ah = first["bbox"]
    bx, by, bw, bh = second["bbox"]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter = max(0.0, min(ax2, bx2) - max(ax, bx)) * max(0.0, min(ay2, by2) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def rectangle(draw: ImageDraw.ImageDraw, bbox: list[float], color: str, width: int = 2) -> None:
    x, y, w, h = bbox
    draw.rectangle((x, y, x + w, y + h), outline=color, width=width)


def label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: str, font: ImageFont.ImageFont) -> None:
    box = draw.textbbox(xy, text, font=font)
    draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=color)
    draw.text(xy, text, fill="black", font=font)


def draw_gt(
    image: Image.Image,
    annotations: list[dict],
    names: dict[int, str],
    focus: dict,
    focus_kind: str,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    small_font = load_font(15)
    for ann in annotations:
        rectangle(draw, ann["bbox"], COLORS[ann["category_id"]])
    rectangle(draw, focus["bbox"], "#fef08a", width=4)
    label(draw, (focus["bbox"][0], max(0, focus["bbox"][1] - 18)), focus_kind, "#fef08a", small_font)
    return result


def draw_predictions(
    image: Image.Image,
    predictions: list[dict],
    annotations: list[dict],
    names: dict[int, str],
    focus: dict,
    focus_kind: str,
) -> tuple[Image.Image, dict | None, float]:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    small_font = load_font(15)
    visible = sorted(
        (pred for pred in predictions if pred["score"] >= PREDICTION_SCORE),
        key=lambda pred: pred["score"],
        reverse=True,
    )[:MAX_PREDICTIONS]
    for pred in visible:
        rectangle(draw, pred["bbox"], COLORS[pred["category_id"]])
        x, y, _, _ = pred["bbox"]
        label(draw, (x, max(0, y - 17)), f"{names[pred['category_id']]} {pred['score']:.2f}", COLORS[pred["category_id"]], small_font)

    car_predictions = [pred for pred in predictions if pred["category_id"] == 1]
    best = max(car_predictions, key=lambda pred: iou(focus, pred), default=None)
    best_iou = iou(focus, best) if best else 0.0
    if best is not None:
        highlight = "#bef264" if best_iou >= 0.5 else "#fb7185"
        rectangle(draw, best["bbox"], highlight, width=4)
        x, y, _, _ = best["bbox"]
        label(draw, (x, max(0, y - 34)), f"{focus_kind} IoU {best_iou:.2f} score {best['score']:.2f}", highlight, small_font)
    return result, best, best_iou


def crop_box(image: Image.Image, focus: dict, margin: int = 90) -> tuple[int, int, int, int]:
    x, y, w, h = focus["bbox"]
    side = max(w, h) + margin * 2
    left = max(0, min(image.width - side, x + w / 2 - side / 2))
    top = max(0, min(image.height - side, y + h / 2 - side / 2))
    return (round(left), round(top), round(left + side), round(top + side))


def make_comparison(
    image: Image.Image,
    sequence: str,
    frame_num: int,
    weather: str,
    annotations: list[dict],
    predictions: list[dict],
    names: dict[int, str],
    focus: dict,
    focus_kind: str,
) -> tuple[Image.Image, float, float]:
    gt = draw_gt(image, annotations, names, focus, focus_kind)
    pred, best_prediction, best_iou = draw_predictions(image, predictions, annotations, names, focus, focus_kind)
    crop = crop_box(image, focus)
    gt_crop = gt.crop(crop).resize((540, 540), Image.Resampling.NEAREST)
    pred_crop = pred.crop(crop).resize((540, 540), Image.Resampling.NEAREST)

    header_height = 64
    footer_height = 590
    canvas = Image.new("RGB", (image.width * 2, header_height + image.height + footer_height), "#111827")
    canvas.paste(gt, (0, header_height))
    canvas.paste(pred, (image.width, header_height))
    canvas.paste(gt_crop, (image.width // 2 - 270, header_height + image.height + 34))
    canvas.paste(pred_crop, (image.width + image.width // 2 - 270, header_height + image.height + 34))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    text_font = load_font(18)
    draw.text((18, 18), f"{sequence} | frame {frame_num} | {weather}", fill="white", font=title_font)
    draw.text((18, header_height + 12), "GROUND TRUTH", fill="white", font=text_font)
    draw.text((image.width + 18, header_height + 12), f"PREDICTIONS score >= {PREDICTION_SCORE:.2f}, top {MAX_PREDICTIONS}", fill="white", font=text_font)
    focus_text = f"{focus_kind}: {focus['bbox'][2]:.1f} x {focus['bbox'][3]:.1f}px | best car IoU={best_iou:.2f}"
    if best_prediction is not None:
        focus_text += f" score={best_prediction['score']:.2f}"
    draw.text((18, header_height + image.height + 8), focus_text, fill="white", font=text_font)
    draw.text((image.width // 2 - 260, header_height + image.height + 554), "FOCUS CROP: GT", fill="white", font=text_font)
    draw.text((image.width + image.width // 2 - 260, header_height + image.height + 554), "FOCUS CROP: PREDICTION", fill="white", font=text_font)
    return canvas, best_iou, float(best_prediction["score"]) if best_prediction else 0.0


def main() -> None:
    args = parse_args()
    with args.annotations.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    with args.predictions.open(encoding="utf-8") as handle:
        predictions = json.load(handle)

    names = {category["id"]: category["name"] for category in coco["categories"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    predictions_by_image: dict[int, list[dict]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_image[prediction["image_id"]].append(prediction)
    images_by_sequence: dict[str, list[dict]] = defaultdict(list)
    for image in coco["images"]:
        images_by_sequence[image["sequence"]].append(image)

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for sequence, images in sorted(images_by_sequence.items()):
        sequence_has_small_car = any(
            ann["category_id"] == 1 and ann["bbox"][2] * ann["bbox"][3] < SMALL_AREA
            for image in images
            for ann in annotations_by_image[image["id"]]
        )

        def selection_key(image: dict) -> tuple[int, int, int]:
            annotations = annotations_by_image[image["id"]]
            small_cars = [
                ann for ann in annotations
                if ann["category_id"] == 1 and ann["bbox"][2] * ann["bbox"][3] < SMALL_AREA
            ]
            car_count = sum(ann["category_id"] == 1 for ann in annotations)
            secondary = len(annotations) if sequence_has_small_car else car_count
            return (len(small_cars), secondary, -image["frame_num"])

        selected = max(images, key=selection_key)
        annotations = annotations_by_image[selected["id"]]
        small_cars = [
            ann for ann in annotations
            if ann["category_id"] == 1 and ann["bbox"][2] * ann["bbox"][3] < SMALL_AREA
        ]
        small_car_count = len(small_cars)
        focus_kind = "focus small car"
        if not small_cars:
            small_cars = [ann for ann in annotations if ann["category_id"] == 1]
            focus_kind = "focus smallest car"
        if not small_cars:
            raise RuntimeError(f"{sequence} has no car annotations")
        focus = min(
            small_cars,
            key=lambda ann: max(
                (iou(ann, pred) for pred in predictions_by_image[selected["id"]] if pred["category_id"] == 1),
                default=0.0,
            ),
        )
        image_path = args.images / selected["file_name"]
        with Image.open(image_path) as source:
            comparison, best_iou, best_score = make_comparison(
                source.convert("RGB"),
                sequence,
                selected["frame_num"],
                selected.get("weather", "unknown"),
                annotations,
                predictions_by_image[selected["id"]],
                names,
                focus,
                focus_kind,
            )
        filename = f"{sequence}_frame_{selected['frame_num']:05d}.png"
        comparison.save(args.output / filename, quality=95)
        rows.append({
            "sequence": sequence,
            "frame_num": selected["frame_num"],
            "weather": selected.get("weather", "unknown"),
            "image": selected["file_name"],
            "gt_boxes": len(annotations),
            "small_car_gt": small_car_count,
            "focus_type": focus_kind.removeprefix("focus ").replace(" ", "_"),
            "focus_bbox": " ".join(f"{value:.1f}" for value in focus["bbox"]),
            "focus_best_car_iou": f"{best_iou:.4f}",
            "focus_best_car_score": f"{best_score:.4f}",
            "comparison": filename,
        })

    with (args.output / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sequence comparisons to {args.output}")


if __name__ == "__main__":
    main()
