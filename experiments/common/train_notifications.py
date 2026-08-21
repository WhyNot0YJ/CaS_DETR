"""Optional Resend notifications for training entry points.

The notification path is deliberately best-effort: a missing or failed email
must never change the training process' exit status.
"""

from __future__ import annotations

import csv
import base64
import functools
import html
import json
import os
import platform
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_RECIPIENT = "yujie8580@gmail.com"
RESEND_EMAIL_ENDPOINT = "https://api.resend.com/emails"
METRIC_LABELS = (
    ("map50", "mAP@0.50"),
    ("map5095", "mAP@0.50:0.95"),
    ("mapsmall50", "AP_small@0.50"),
    ("mapsmall5095", "AP_small@0.50:0.95"),
)


def _load_repo_env() -> None:
    """Load simple KEY=VALUE entries from the repository .env if present.

    Existing process variables win, so CI/launcher configuration remains
    authoritative.  This intentionally avoids adding a dotenv dependency.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


_load_repo_env()


def _normalise_key(value: object) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _as_metric(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _metrics_from_row(row: Mapping[str, object]) -> Dict[str, float]:
    values = {_normalise_key(k): v for k, v in row.items()}
    aliases = {
        "map50": (
            "map50",
            "map_50",
            "map_0_5",
            "metrics_map50_b",
            "metrics_map50",
            "val_map_50",
            "val_map50",
            "coco_eval_bbox_1",
        ),
        "map5095": (
            "map5095",
            "map_5095",
            "map_0_5_0_95",
            "metrics_map50_95_b",
            "metrics_map50_95",
            "val_map_50_95",
            "val_map5095",
            "coco_eval_bbox_0",
        ),
        "mapsmall50": (
            "mapsmall50",
            "map_small50",
            "map_small_50",
            "ap_small50",
            "ap_small_50",
        ),
        "mapsmall5095": (
            "mapsmall5095",
            "map_small5095",
            "map_small_5095",
            "ap_small5095",
            "ap_small_5095",
            "ap_small",
            "map_s",
            "coco_eval_bbox_3",
        ),
    }
    result: Dict[str, float] = {}
    for metric, candidates in aliases.items():
        for candidate in candidates:
            value = _as_metric(values.get(candidate))
            if value is not None:
                result[metric] = value
                break
    return result


def _read_csv_metrics(
    path: Path,
    *,
    run_id: Optional[str] = None,
    require_run_id: bool = False,
) -> Dict[str, float]:
    if not path.is_file() or not path.stat().st_size:
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return {}
    if run_id:
        matching = [row for row in rows if str(row.get("run_id", "")) == run_id]
        if matching:
            rows = matching
        elif require_run_id:
            return {}
    return _metrics_from_row(rows[-1]) if rows else {}


def _read_detr_log_metrics(path: Path) -> Dict[str, float]:
    if not path.is_file():
        return {}
    latest: Dict[str, float] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return latest
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if "coco_eval_bbox" not in str(key):
                continue
            if not isinstance(value, (list, tuple)):
                continue
            latest.update(_metrics_from_row({f"coco_eval_bbox_{i}": item for i, item in enumerate(value)}))
    return latest


def _loss_from_row(row: Mapping[str, object], prefixes: Sequence[str]) -> float:
    normalized = {
        str(key).lower().replace("/", "_").replace("-", "_"): raw
        for key, raw in row.items()
    }
    total_keys = [f"{prefix}_loss" for prefix in prefixes]
    total_keys += [f"{prefix}_total_loss" for prefix in prefixes]
    for key in total_keys:
        value = _as_metric(normalized.get(key))
        if value is not None:
            return value
    return sum(
        value
        for key, raw in normalized.items()
        if any(key.startswith(f"{prefix}_") for prefix in prefixes)
        and "loss" in key
        and (value := _as_metric(raw)) is not None
    )


def collect_training_metrics(output_dir: Optional[Path | str]) -> Tuple[Dict[str, float], str]:
    """Read metrics produced by the current run, without inventing missing values."""
    if not output_dir:
        return {}, ""
    output = Path(output_dir).expanduser()
    if not output.exists():
        return {}, ""

    metrics: Dict[str, float] = {}
    source: list[str] = []
    for csv_path in sorted(output.rglob("*.csv"), key=lambda p: p.stat().st_mtime):
        values = _read_csv_metrics(csv_path)
        if values:
            metrics.update(values)
            source.append(str(csv_path))
    log_values = _read_detr_log_metrics(output / "log.txt")
    if log_values:
        for key, value in log_values.items():
            metrics.setdefault(key, value)
        source.append(str(output / "log.txt"))

    protocol = os.environ.get("EXPERIMENT_DATASET_PROTOCOL", "dairv2x").lower()
    report_root = os.environ.get("EXPERIMENT_RESULTS_DIR")
    if report_root:
        report_csv = Path(report_root).expanduser() / "eval_metrics.csv"
    else:
        report_csv = Path(__file__).resolve().parents[1] / "reports" / protocol / "eval_metrics.csv"
    report_values = _read_csv_metrics(report_csv, run_id=output.name, require_run_id=True)
    if report_values:
        metrics.update(report_values)
        source.append(str(report_csv))
    return metrics, ", ".join(dict.fromkeys(source))


def _read_training_history(output_dir: Path) -> Dict[str, List[float]]:
    """Read the small common subset needed for a training-curve figure."""
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "map50": [],
        "map5095": [],
    }

    log_path = output_dir / "log.txt"
    if log_path.is_file():
        try:
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError, json.JSONDecodeError):
            rows = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            epoch = _as_metric(row.get("epoch"))
            if epoch is None:
                continue
            train_loss = _loss_from_row(row, ("train",))
            val_loss = _loss_from_row(row, ("test", "val", "valid", "validation"))
            values = row.get("test_coco_eval_bbox")
            eval_metrics = (
                _metrics_from_row({f"coco_eval_bbox_{i}": item for i, item in enumerate(values)})
                if isinstance(values, (list, tuple))
                else {}
            )
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["map50"].append(eval_metrics.get("map50", float("nan")))
            history["map5095"].append(eval_metrics.get("map5095", float("nan")))
        if history["epoch"]:
            return history

    csv_paths = sorted(output_dir.rglob("results.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_paths:
        return history
    try:
        with csv_paths[-1].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return history
    for index, row in enumerate(rows):
        epoch = _as_metric(row.get("epoch"))
        if epoch is None:
            epoch = float(index)
        train_loss = _loss_from_row(row, ("train",))
        val_loss = _loss_from_row(row, ("test", "val", "valid", "validation"))
        values = _metrics_from_row(row)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["map50"].append(values.get("map50", float("nan")))
        history["map5095"].append(values.get("map5095", float("nan")))
    return history


def _generate_report_images(
    output_dir: Optional[Path], metrics: Mapping[str, object]
) -> List[Path]:
    """Create compact email-safe plots; plotting failures never fail training."""
    if output_dir is None or not output_dir.is_dir():
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[train-notify] plots skipped: matplotlib/numpy unavailable", flush=True)
        return []

    generated: List[Path] = []
    history = _read_training_history(output_dir)
    epochs = np.asarray(history["epoch"], dtype=float)
    if epochs.size:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle("Training Report", fontsize=14, fontweight="bold")
        for key, label, color in (
            ("train_loss", "Train loss", "tab:blue"),
            ("val_loss", "Validation loss", "tab:red"),
        ):
            values = np.asarray(history[key], dtype=float)
            if np.isfinite(values).any() and np.nanmax(values) > 0:
                axes[0].plot(epochs, values, label=label, color=color, linewidth=2)
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
        for key, label, color in (
            ("map50", "mAP@0.50", "tab:green"),
            ("map5095", "mAP@0.50:0.95", "tab:purple"),
        ):
            values = np.asarray(history[key], dtype=float)
            if np.isfinite(values).any():
                axes[1].plot(epochs, values, label=label, color=color, linewidth=2)
        axes[1].set_title("Validation AP")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(bottom=0)
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        fig.tight_layout()
        path = output_dir / "email_training_report.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)

    available = [(key, label, _as_metric(metrics.get(key))) for key, label in METRIC_LABELS]
    available = [(key, label, value) for key, label, value in available if value is not None]
    if available:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        bars = axis.bar([label for _, label, _ in available], [value for _, _, value in available], color=[
            "#2ca02c", "#9467bd", "#ff7f0e", "#d62728",
        ][:len(available)])
        axis.set_title("Final Metrics")
        axis.set_ylim(0, max(1.0, max(value for _, _, value in available) * 1.2))
        axis.tick_params(axis="x", labelrotation=18)
        axis.grid(axis="y", alpha=0.25)
        for bar, (_, _, value) in zip(bars, available):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
        fig.tight_layout()
        path = output_dir / "email_metrics_summary.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)
    return generated


def _select_attachments(output_dir: Optional[Path], generated: Sequence[Path]) -> List[Path]:
    if output_dir is None or not output_dir.is_dir():
        return [path for path in generated if path.is_file()]
    selected: List[Path] = []
    training_curve = output_dir / "training_curves.png"
    candidates = [training_curve] if training_curve.is_file() else list(generated)
    candidates += [output_dir / "email_metrics_summary.png"]
    for pattern in (
        "vis_train_end/*.jpg",
        "vis_train_end/*.png",
        "vis_cass_importance/*.jpg",
        "vis_cass_importance/*.png",
        "val_batch*_pred.jpg",
        "val_batch*_labels.jpg",
    ):
        candidates.extend(sorted(output_dir.glob(pattern)))
    seen = set()
    total_bytes = 0
    for path in candidates:
        path = Path(path)
        if not path.is_file() or path in seen or path.stat().st_size > 5 * 1024 * 1024:
            continue
        if total_bytes + path.stat().st_size > 8 * 1024 * 1024 or len(selected) >= 4:
            break
        seen.add(path)
        selected.append(path)
        total_bytes += path.stat().st_size
    return selected


def _format_metrics(metrics: Mapping[str, object]) -> str:
    return "\n".join(
        f"{label}: {float(metrics[key]):.4f}" if key in metrics else f"{label}: N/A"
        for key, label in METRIC_LABELS
    )


def _html_metrics(metrics: Mapping[str, object]) -> str:
    rows = []
    for key, label in METRIC_LABELS:
        value = f"{float(metrics[key]):.4f}" if key in metrics else "N/A"
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{html.escape(value)}</td></tr>")
    return "<table border='1' cellpadding='6' cellspacing='0'>" + "".join(rows) + "</table>"


def _send_email(
    *,
    subject: str,
    text_body: str,
    html_body: str,
    attachments: Optional[Sequence[Path]] = None,
) -> bool:
    enabled = os.environ.get("TRAIN_NOTIFY_ENABLED", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        print("[train-notify] skipped: TRAIN_NOTIFY_ENABLED is disabled", flush=True)
        return False
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    sender = os.environ.get("RESEND_FROM", "onboarding@resend.dev").strip()
    recipient = os.environ.get("RESEND_TO", DEFAULT_RECIPIENT).strip()
    if not api_key:
        print("[train-notify] skipped: RESEND_API_KEY is required", flush=True)
        return False
    body: Dict[str, object] = {
        "from": sender,
        "to": [recipient],
        "subject": subject,
        "text": text_body,
        "html": html_body,
    }
    encoded_attachments = []
    for path in attachments or ():
        try:
            encoded_attachments.append(
                {"filename": path.name, "content": base64.b64encode(path.read_bytes()).decode("ascii")}
            )
        except OSError as exc:
            print(f"[train-notify] attachment skipped: {path}: {exc}", flush=True)
    if encoded_attachments:
        body["attachments"] = encoded_attachments
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        RESEND_EMAIL_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "CaS-DETR-training-notifier/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"HTTP {response.status}: {response.read().decode('utf-8', 'replace')[:500]}")
        print(f"[train-notify] sent: {recipient}", flush=True)
        return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        print(f"[train-notify] send failed: HTTP {exc.code}: {detail}", flush=True)
        return False
    except (OSError, urllib.error.URLError, RuntimeError) as exc:
        print(f"[train-notify] send failed: {type(exc).__name__}: {exc}", flush=True)
        return False


def _is_primary_process() -> bool:
    """Avoid duplicate notifications when a CLI is launched by torchrun."""
    for name in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            try:
                return int(raw) == 0
            except ValueError:
                return True
    return True


def _result_info(result: object) -> Tuple[Optional[Path], Dict[str, float]]:
    if not isinstance(result, Mapping):
        return None, {}
    output = result.get("output_dir")
    output_path = Path(str(output)).expanduser() if output else None
    raw_metrics = result.get("metrics") or {}
    metrics = dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {}
    return output_path, {key: value for key, value in metrics.items() if _as_metric(value) is not None}


def run_with_training_notification(
    function: Callable[[], Any],
    *,
    experiment: str,
    config_path: Optional[str] = None,
    mode: str = "训练",
    output_dir: Optional[str | Path] = None,
) -> Any:
    """AOP wrapper: notify on both success and failure, then preserve the result/exception."""
    started = time.monotonic()
    try:
        result = function()
    except BaseException as exc:
        elapsed = time.monotonic() - started
        details = f"{type(exc).__name__}: {exc}".strip()
        traceback_text = traceback.format_exc(limit=12)
        body = (
            f"实验状态：失败\n实验名称：{experiment}\n执行类型：{mode}\n"
            f"耗时：{elapsed:.1f}s\n配置：{config_path or 'N/A'}\n"
            f"输出目录：{output_dir or 'N/A'}\n失败原因：{details}\n\n"
            f"Traceback（截取）：\n{traceback_text[-6000:]}"
        )
        failure_output = Path(output_dir).expanduser() if output_dir else None
        failure_metrics, _ = collect_training_metrics(failure_output)
        failure_plots = _generate_report_images(failure_output, failure_metrics)
        if _is_primary_process():
            _send_email(
                subject=f"{experiment} [失败]",
                text_body=body,
                html_body="<pre>" + html.escape(body) + "</pre>",
                attachments=_select_attachments(failure_output, failure_plots),
            )
        raise

    result_output, result_metrics = _result_info(result)
    resolved_output = result_output or (Path(output_dir) if output_dir else None)
    collected, source = collect_training_metrics(resolved_output)
    metrics = {**collected, **result_metrics}
    report_plots = _generate_report_images(resolved_output, metrics)
    attachments = _select_attachments(resolved_output, report_plots)
    elapsed = time.monotonic() - started
    metric_text = _format_metrics(metrics)
    body = (
        f"实验状态：成功\n实验名称：{experiment}\n执行类型：{mode}\n"
        f"耗时：{elapsed:.1f}s\n配置：{config_path or 'N/A'}\n"
        f"输出目录：{resolved_output or 'N/A'}\n指标来源：{source or '未找到评估结果'}\n\n"
        f"{metric_text}\n\n主机：{platform.node()}\nPython：{sys.version.split()[0]}"
    )
    if _is_primary_process():
        _send_email(
            subject=f"{experiment} [成功]",
            text_body=body,
            html_body=(
                f"<h3>实验成功</h3><p><b>实验：</b>{html.escape(experiment)}</p>"
                f"<p><b>类型：</b>{html.escape(mode)}<br><b>输出：</b>{html.escape(str(resolved_output or 'N/A'))}</p>"
                f"{_html_metrics(metrics)}<p>耗时：{elapsed:.1f}s</p>"
            ),
            attachments=attachments,
        )
    return result


def notify_training_entry(framework: str) -> Callable:
    """Decorate a CLI ``main`` and notify without changing its exception behavior."""

    def decorator(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            arg_object = args[0] if args else None
            config_path = getattr(arg_object, "config", None)
            output_dir = getattr(arg_object, "output_dir", None)
            test_only = bool(getattr(arg_object, "test_only", False))
            argv = sys.argv[1:]
            for index, token in enumerate(argv[:-1]):
                if token in {"-c", "--config"} and not config_path:
                    config_path = argv[index + 1]
                elif token in {"--output-dir"} and not output_dir:
                    output_dir = argv[index + 1]
            for token in argv:
                if token.startswith("--config=") and not config_path:
                    config_path = token.split("=", 1)[1]
                if token == "--test-only":
                    test_only = True

            stem = Path(str(config_path)).stem if config_path else "training"
            experiment = f"{framework}/{stem}"
            return run_with_training_notification(
                lambda: function(*args, **kwargs),
                experiment=experiment,
                config_path=str(config_path) if config_path else None,
                mode="验证" if test_only else "训练",
                output_dir=output_dir,
            )

        return wrapped

    return decorator
