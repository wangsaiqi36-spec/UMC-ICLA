"""Lightweight, opt-in efficiency logging for UMC-ICLA experiments.

The recorder only stores measurements.  It deliberately does not run extra
forward passes, evaluations, or model analysis, so enabling it cannot change
the experiment protocol.
"""

from __future__ import absolute_import

import contextlib
import datetime
import json
import os
import platform
import subprocess
import time
import uuid

import torch


SCHEMA_VERSION = "1.0"


def _utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _json_safe(value):
    """Convert common config values to strict-JSON-compatible objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError, RuntimeError):
            pass
    return str(value)


def _git_state():
    result = {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        result = {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


class EfficiencyRecorder(object):
    """Persist raw timing, CUDA-memory, parameter, and metric measurements."""

    def __init__(self, args, data=None):
        self.enabled = bool(getattr(args, "efficiency_enabled", False))
        self.protocol_id = getattr(args, "efficiency_protocol_id", "umc-icla-efficiency-v1")
        self.run_id = None
        self.run_dir = None
        self._wall_start = None
        self._segment_index = 0
        self._epoch_index = 0
        self._logging_seconds = 0.0
        self._segments = []
        self._active_segments = []
        self._epochs = []
        self._evaluations = []
        self._parameter_snapshots = []
        self._status = "disabled"
        self._stop_reason = None
        self._args = args

        if not self.enabled:
            return

        # uuid.uuid4 uses operating-system entropy and does not consume any of
        # the experiment's Python, NumPy, or PyTorch random-number streams.
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.run_id = "%s-%s" % (stamp, uuid.uuid4().hex[:8])
        output_root = os.path.abspath(getattr(args, "efficiency_output_dir", "efficiency_logs"))
        self.run_dir = os.path.join(
            output_root, str(args.method), str(args.dataset), self.run_id
        )
        os.makedirs(self.run_dir, exist_ok=False)
        self._wall_start = time.perf_counter()
        self._status = "running"

        train_size = getattr(args, "num_train_examples", None)
        if train_size is None and data is not None:
            try:
                train_size = len(data.mm_data["train"])
            except (AttributeError, KeyError, TypeError):
                train_size = None

        config = {}
        for key, value in dict(args).items():
            if key != "efficiency_recorder":
                config[key] = _json_safe(value)

        cuda_available = torch.cuda.is_available()
        cuda_devices = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                cuda_devices.append({
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                })

        cudnn_version = None
        if torch.backends.cudnn.is_available():
            cudnn_version = torch.backends.cudnn.version()

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "method": getattr(args, "method", None),
            "model_variant": getattr(args, "model_variant", None),
            "multimodal_method": getattr(args, "multimodal_method", None),
            "dataset": getattr(args, "dataset", None),
            "seed": getattr(args, "seed", None),
            "protocol_id": self.protocol_id,
            "started_at_utc": _utc_now(),
            "code": _git_state(),
            "configuration": config,
            "data": {
                "train_samples": train_size,
                "test_samples": self._dataset_length(data, "test"),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor() or None,
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": cudnn_version,
                "cuda_available": cuda_available,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_devices": cuda_devices,
            },
            "measurement_scope": {
                "parameter_scope": "all parameters in the online model; buffers are excluded",
                "training_time": (
                    "all pretrain/main-train epoch work, including data loading and transfer, "
                    "ICLA augmentation/virtual contrast, feature extraction, clustering, sample "
                    "selection, forward/backward, and optimizer steps"
                ),
                "inference_time": (
                    "end-to-end final test: feature extraction, KMeans fitting/prediction, and metric computation"
                ),
                "memory": (
                    "per-segment PyTorch CUDA allocator peaks (allocated and reserved); "
                    "not whole-device process memory"
                ),
                "excluded_from_training_time": "checkpoint writes, final test, result logging, and setup",
                "wall_scope": "recorder initialization through completion/failure finalization",
            },
        }
        self._write_json("metadata.json", metadata)
        self._write_json("parameters.json", self._parameters_document())
        self._write_summary()

    @staticmethod
    def _dataset_length(data, split):
        if data is None:
            return None
        try:
            return len(data.mm_data[split])
        except (AttributeError, KeyError, TypeError):
            return None

    @property
    def logging_seconds(self):
        return self._logging_seconds

    def _write_json(self, filename, value):
        if not self.enabled:
            return
        started = time.perf_counter()
        path = os.path.join(self.run_dir, filename)
        temporary_path = path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, path)
        self._logging_seconds += time.perf_counter() - started

    def _append_jsonl(self, filename, value):
        if not self.enabled:
            return
        started = time.perf_counter()
        path = os.path.join(self.run_dir, filename)
        with open(path, "a", encoding="utf-8") as handle:
            json.dump(_json_safe(value), handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
        self._logging_seconds += time.perf_counter() - started

    def _cuda_device(self):
        device = getattr(self._args, "device", None)
        if not torch.cuda.is_available() or device is None:
            return None
        device = torch.device(device)
        return device if device.type == "cuda" else None

    @contextlib.contextmanager
    def segment(self, stage, kind, epoch_stage=None, details=None):
        """Measure a non-overlapping compute segment and append it to JSONL."""
        if not self.enabled:
            yield None
            return
        record = self.begin_segment(stage, kind, epoch_stage, details)
        try:
            yield record
        except BaseException as error:
            self.end_segment(record, status="failed", error=error)
            raise
        else:
            self.end_segment(record, status="completed")

    def begin_segment(self, stage, kind, epoch_stage=None, details=None):
        """Start a measured segment for call sites where a context is intrusive."""
        if not self.enabled:
            return None
        self._segment_index += 1
        device = self._cuda_device()
        if device is not None:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "segment_id": "segment-%06d" % self._segment_index,
            "stage": stage,
            "epoch_stage": epoch_stage,
            "kind": kind,
            "status": "running",
            "details": _json_safe(details or {}),
            "_started_perf": time.perf_counter(),
            "_device": device,
        }
        self._active_segments.append(record)
        return record

    def end_segment(self, record, status="completed", error=None):
        """Finish a segment returned by :meth:`begin_segment`."""
        if not self.enabled or record is None:
            return record
        if "_started_perf" not in record:
            return record
        device = record.pop("_device")
        started = record.pop("_started_perf")
        if device is not None:
            torch.cuda.synchronize(device)
        record["status"] = status
        if error is not None:
            record["error_type"] = type(error).__name__
            record["error_message"] = str(error)
        record["wall_seconds"] = time.perf_counter() - started
        record["peak_allocated_bytes"] = (
            int(torch.cuda.max_memory_allocated(device)) if device is not None else None
        )
        record["peak_reserved_bytes"] = (
            int(torch.cuda.max_memory_reserved(device)) if device is not None else None
        )
        self._segments.append(record.copy())
        if record in self._active_segments:
            self._active_segments.remove(record)
        self._append_jsonl("segments.jsonl", record)
        return record

    def record_epoch(self, stage, epoch_stage, segment, loss_components,
                     samples_seen, optimizer_steps, learning_rates,
                     loss_reduction="sample_weighted_mean", loss_total_mean=None):
        if not self.enabled or segment is None:
            return
        self._epoch_index += 1
        train_segments = [item for item in self._segments if item["kind"] == "train"]
        cumulative_seconds = sum(item["wall_seconds"] for item in train_segments)
        losses = {key: _json_safe(value) for key, value in loss_components.items()}
        present_losses = [value for value in losses.values() if isinstance(value, (int, float))]
        if loss_total_mean is None:
            loss_total_mean = sum(present_losses) if present_losses else None
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "stage": stage,
            "epoch_stage": int(epoch_stage),
            "epoch_global": self._epoch_index,
            "segment_ids": [segment["segment_id"]],
            "train_seconds": segment["wall_seconds"],
            "cumulative_train_seconds": cumulative_seconds,
            "peak_allocated_bytes": segment["peak_allocated_bytes"],
            "peak_reserved_bytes": segment["peak_reserved_bytes"],
            "loss_total_mean": _json_safe(loss_total_mean),
            "loss_components": losses,
            "loss_reduction": loss_reduction,
            "learning_rates": [float(value) for value in learning_rates],
            "samples_seen": int(samples_seen),
            "optimizer_steps": int(optimizer_steps),
            "status": segment["status"],
        }
        self._epochs.append(record)
        self._append_jsonl("epochs.jsonl", record)
        self._write_summary()

    def record_evaluation(self, stage, segment, metrics, split, n_samples,
                          role="final"):
        if not self.enabled or segment is None:
            return
        train_seconds = sum(
            item["wall_seconds"] for item in self._segments if item["kind"] == "train"
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "state_id": "%s-after-training" % stage,
            "stage": stage,
            "epoch_stage": None,
            "epoch_global": self._epoch_index,
            "cumulative_train_seconds": train_seconds,
            "eval_seconds": segment["wall_seconds"],
            "segment_id": segment["segment_id"],
            "split": split,
            "n_samples": int(n_samples) if n_samples is not None else None,
            "metrics": {key: _json_safe(value) for key, value in metrics.items()},
            "metric_unit": "percent (as returned by utils.metrics.clustering_score)",
            "role": role,
            "status": segment["status"],
        }
        self._evaluations.append(record)
        self._append_jsonl("evaluations.jsonl", record)
        self._write_summary()

    @staticmethod
    def _parameter_counts(parameters):
        seen = set()
        total = 0
        trainable = 0
        parameter_bytes = 0
        for parameter in parameters:
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            count = int(parameter.numel())
            total += count
            parameter_bytes += count * int(parameter.element_size())
            if parameter.requires_grad:
                trainable += count
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
            "parameter_bytes": parameter_bytes,
        }

    def record_parameters(self, model, stage, paper_snapshot=False):
        if not self.enabled:
            return
        target = model.module if hasattr(model, "module") else model
        counts = self._parameter_counts(target.parameters())
        by_module = {}
        for name, module in target.named_modules():
            if not name or name.count(".") > 1:
                continue
            module_counts = self._parameter_counts(module.parameters())
            if module_counts["total_params"]:
                by_module[name] = module_counts
        snapshot = {
            "snapshot_id": "parameters-%03d" % (len(self._parameter_snapshots) + 1),
            "stage": stage,
            "captured_at_utc": _utc_now(),
            "paper_snapshot": bool(paper_snapshot),
            "scope": "complete online model, including text encoder, fusion backbone, and projection heads",
            "module_note": "module entries are descriptive and may overlap; use top-level totals for comparisons",
            "by_module": by_module,
        }
        snapshot.update(counts)
        self._parameter_snapshots.append(snapshot)
        self._write_json("parameters.json", self._parameters_document())
        self._write_summary()

    def _parameters_document(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "snapshots": self._parameter_snapshots,
        }

    def _summary_document(self):
        train_segments = [item for item in self._segments if item["kind"] == "train"]
        completed_train = [item for item in train_segments if item["status"] == "completed"]
        time_by_stage = {}
        peak_by_stage = {}
        for item in completed_train:
            stage = item["stage"]
            time_by_stage[stage] = time_by_stage.get(stage, 0.0) + item["wall_seconds"]
            allocated = item.get("peak_allocated_bytes")
            reserved = item.get("peak_reserved_bytes")
            current = peak_by_stage.setdefault(stage, {
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            })
            if allocated is not None:
                current["peak_allocated_bytes"] = max(current["peak_allocated_bytes"] or 0, allocated)
            if reserved is not None:
                current["peak_reserved_bytes"] = max(current["peak_reserved_bytes"] or 0, reserved)

        def total_for(kind):
            return sum(
                item["wall_seconds"] for item in self._segments
                if item["kind"] == kind and item["status"] == "completed"
            )

        def peak_for(field):
            values = [item[field] for item in completed_train if item.get(field) is not None]
            return max(values) if values else None

        epochs_by_stage = {}
        steps_by_stage = {}
        for epoch in self._epochs:
            stage = epoch["stage"]
            epochs_by_stage[stage] = epochs_by_stage.get(stage, 0) + 1
            steps_by_stage[stage] = steps_by_stage.get(stage, 0) + epoch["optimizer_steps"]

        final_evaluation = self._evaluations[-1] if self._evaluations else None
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "method": getattr(self._args, "method", None),
            "model_variant": getattr(self._args, "model_variant", None),
            "dataset": getattr(self._args, "dataset", None),
            "seed": getattr(self._args, "seed", None),
            "protocol_id": self.protocol_id,
            "status": self._status,
            "stop_reason": self._stop_reason,
            "completed_epochs_by_stage": epochs_by_stage,
            "optimizer_steps_by_stage": steps_by_stage,
            "parameter_snapshot_ids": [item["snapshot_id"] for item in self._parameter_snapshots],
            "train_seconds": sum(item["wall_seconds"] for item in completed_train),
            "time_by_stage_seconds": time_by_stage,
            "train_peak_allocated_bytes": peak_for("peak_allocated_bytes"),
            "train_peak_reserved_bytes": peak_for("peak_reserved_bytes"),
            "train_peak_by_stage": peak_by_stage,
            "inference_seconds": total_for("inference"),
            "evaluation_seconds": total_for("eval"),
            "checkpoint_seconds": total_for("checkpoint"),
            "logging_seconds": self._logging_seconds,
            "wall_seconds": (
                time.perf_counter() - self._wall_start if self._wall_start is not None else None
            ),
            "wall_scope": "recorder initialization through the latest summary update",
            "final_evaluation": final_evaluation,
        }

    def _write_summary(self):
        if self.enabled:
            self._write_json("run_summary.json", self._summary_document())

    def finalize(self, status="completed", stop_reason=None):
        if not self.enabled:
            return
        if status != "completed":
            for segment in list(self._active_segments):
                self.end_segment(
                    segment,
                    status="failed",
                    error=RuntimeError(stop_reason or "run failed"),
                )
        self._status = status
        self._stop_reason = stop_reason
        self._write_summary()


class NullEfficiencyRecorder(EfficiencyRecorder):
    """Compatibility placeholder used only for explicit construction in tests."""

    def __init__(self):
        self.enabled = False
