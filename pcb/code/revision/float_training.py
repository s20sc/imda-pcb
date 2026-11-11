#!/usr/bin/env python3
"""NEW v2 float-contract streaming trainer, CPU/CUDA FP32, matched information controls.

No legacy checkpoint import or test inference. Allocation is frozen before NPZ.
The stateless sampling plan uses seed/update and is identical across input modes.
Only final-resolution masked linear-RGB L1 is optimized; no legacy FFT claim.
Default invocation validates files only; explicit --write creates a new run.
"""
import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ALLOCATION_SCHEMA = "imda-float-allocation-v2"
CHECKPOINT_SCHEMA = "imda-float-training-checkpoint-v3"
MODES = ("full_maps", "rgb_zero_maps", "endpoint_only")


def _load_io():
    spec = importlib.util.spec_from_file_location("imda_v3_float_io", HERE/"float_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


float_io = _load_io()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def code_hashes():
    names = ("float_training.py", "float_dataset.py", "GAMD/models/__init__.py",
             "GAMD/models/GAMDNet.py", "GAMD/models/layers.py")
    return {name: file_hash(HERE/name) for name in names}


def _strict_json(path):
    def reject(value):
        raise ValueError("Non-finite JSON constant: "+value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: "+key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=pairs)


def prepare_allocation(manifest_path):
    """Stream one file at a time; keep metadata/paths/hashes, never the image pool."""
    path = Path(manifest_path).resolve()
    manifest = _strict_json(path)
    if set(manifest) != {"allocation_schema", "allocation_id", "samples"} or manifest["allocation_schema"] != ALLOCATION_SCHEMA:
        raise ValueError("Explicit v2 allocation required; old scene semantics are not migrated")
    if not isinstance(manifest["allocation_id"], str) or not manifest["allocation_id"].strip() or not isinstance(manifest["samples"], list):
        raise ValueError("Nonempty allocation_id and explicit samples required")
    allocation_hash = file_hash(path)
    fields = set(float_io.PROVENANCE_FIELDS)-{"split_manifest_sha256"}
    seen_ids, seen_paths, groups = set(), set(), {}
    splits, records, scales = {x: [] for x in ("train", "valid", "test")}, [], set()
    maximum_sample_array_bytes = 0
    for entry in manifest["samples"]:
        if not isinstance(entry, dict) or set(entry) != fields | {"npz"}:
            raise ValueError("Allocation entries must have exact v2 provenance plus relative npz; no NPZ hashes")
        provenance = {key: entry[key] for key in fields}
        provenance["split_manifest_sha256"] = allocation_hash
        float_io.validate_provenance(provenance)
        split, identifier = entry["split_name"], entry["sample_id"]
        if split not in splits or identifier in seen_ids:
            raise ValueError("Duplicate sample_id or invalid split_name")
        if not isinstance(entry["npz"], str) or not entry["npz"].strip():
            raise ValueError("A relative NPZ path is required")
        relative = Path(entry["npz"])
        sample_path = (path.parent/relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not sample_path.is_relative_to(path.parent) or sample_path in seen_paths:
            raise ValueError("NPZ traversal, external symlink or duplicate path")
        seen_ids.add(identifier)
        seen_paths.add(sample_path)
        dataset = (entry["source_dataset_id"], entry["source_dataset_version"])
        keys = [("image", *dataset, entry["source_image_id"]),
                ("declared_group", *dataset, entry["group_kind"], entry["group_id"]),
                ("trajectory_family", entry["trajectory_family_id"]),
                ("trajectory", entry["trajectory_id"])]
        for key in ("physical_scene_id", "physical_session_id"):
            if entry[key] is not None:
                keys.append((key, *dataset, entry[key]))
        for key in keys:
            if key in groups and groups[key] != split:
                raise ValueError("Known source/group/trajectory crosses split: "+repr(key))
            groups[key] = split
        before = file_hash(sample_path)
        sample = float_io.load_npz_sample(sample_path, expected_split=split, require_target=True,
                                         expected_manifest_sha256=allocation_hash)
        if file_hash(sample_path) != before or sample["metadata"]["provenance"] != provenance:
            raise ValueError("NPZ changed or provenance differs from frozen allocation")
        height, width = sample["network_input_9"].shape[:2]
        if height % 4 or width % 4 or not sample["valid_mask"].any():
            raise ValueError("Plus needs H/W divisible by 4 and a nonempty availability mask")
        scales.add(sample["metadata"]["map_scale_px"])
        maximum_sample_array_bytes = max(maximum_sample_array_bytes,
                                        sum(a.nbytes for a in sample.values() if isinstance(a, np.ndarray)))
        record = {"sample_id": identifier, "split_name": split, "npz": entry["npz"],
                  "sha256": before, "height": height, "width": width,
                  "valid_pixels": int(sample["valid_mask"].sum()), "provenance": provenance}
        splits[split].append(record)
        records.append(record)
        del sample
    if any(not splits[name] for name in splits) or len(scales) != 1:
        raise ValueError("Nonempty train/valid/test and one explicit map scale required")
    if file_hash(path) != allocation_hash:
        raise ValueError("Allocation changed during preflight")
    return {"allocation": manifest, "allocation_path": str(path), "allocation_sha256": allocation_hash,
            "splits": splits, "npz_records": records, "map_scale_px": next(iter(scales)),
            "maximum_sample_array_bytes": maximum_sample_array_bytes}


def build_plus(num_res=20, device="cpu"):
    if type(num_res) is not int or num_res < 1:
        raise ValueError("Positive num_res required; 1 is an explicitly labeled engineering smoke")
    name = "imda_v3_float_models"
    if name not in sys.modules:
        directory = HERE/"GAMD/models"
        spec = importlib.util.spec_from_file_location(name, directory/"__init__.py", submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    network = importlib.import_module(name+".GAMDNet")
    with contextlib.redirect_stdout(io.StringIO()):
        return network.GAMDUNetPlus(num_res=num_res).to(device)


def _read_record(prepared, record):
    path = Path(prepared["allocation_path"]).parent/record["npz"]
    if file_hash(path) != record["sha256"]:
        raise ValueError("NPZ changed since preflight")
    sample = float_io.load_npz_sample(path, expected_split=record["split_name"], require_target=True,
                                     expected_manifest_sha256=prepared["allocation_sha256"])
    if file_hash(path) != record["sha256"] or sample["metadata"]["provenance"] != record["provenance"]:
        raise ValueError("NPZ changed while loading")
    return sample


def sample_plan(prepared, seed, update, batch_size, patch_size):
    """Stateless with-replacement plan; crop and IDs independent of input mode."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, update, 0x1ADA])))
    records = prepared["splits"]["train"]
    plan = []
    for index in rng.integers(0, len(records), size=batch_size):
        record = records[int(index)]
        height, width = record["height"], record["width"]
        ph, pw = (patch_size, patch_size) if patch_size else (height, width)
        if ph > height or pw > width:
            raise ValueError("Patch exceeds a training source; no resizing or retry sampling")
        top, left = int(rng.integers(height-ph+1)), int(rng.integers(width-pw+1))
        plan.append((int(index), top, left, ph, pw))
    if len({(p[3], p[4]) for p in plan}) != 1:
        raise ValueError("Uncropped batch has varying shapes; choose an explicit common patch")
    return plan


def _apply_mode(batch, input_mode):
    if input_mode not in MODES:
        raise ValueError("Unknown information-control mode")
    batch["input"] = batch["input"].clone()
    if input_mode == "rgb_zero_maps":
        batch["input"][:, 3:] = 0
    elif input_mode == "endpoint_only":
        batch["input"][:, 3:6] = 0
    return batch


def training_batch(prepared, plan, input_mode, device):
    inputs, targets, masks = [], [], []
    for index, top, left, height, width in plan:
        sample = _read_record(prepared, prepared["splits"]["train"][index])
        crop = np.s_[top:top+height, left:left+width]
        inputs.append(sample["network_input_9"][crop].copy())
        targets.append(sample["sharp_linear_rgb"][crop].copy())
        masks.append(sample["valid_mask"][crop].copy())
        del sample
    batch = {"input": torch.from_numpy(np.stack(inputs).transpose(0, 3, 1, 2).copy()).to(device),
             "target": torch.from_numpy(np.stack(targets).transpose(0, 3, 1, 2).copy()).to(device),
             "valid_mask": torch.from_numpy(np.stack(masks)[:, None]).to(device)}
    if not batch["valid_mask"].flatten(1).any(dim=1).all():
        raise ValueError("A planned crop has no valid pixels; no outcome-based resampling")
    return _apply_mode(batch, input_mode)


def masked_spatial_l1(prediction, target, valid_mask):
    if prediction.shape != target.shape or valid_mask.shape != (prediction.shape[0], 1, *prediction.shape[2:]):
        raise ValueError("Prediction/target/mask shape mismatch")
    if valid_mask.dtype != torch.bool or not valid_mask.flatten(1).any(dim=1).all() or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Finite values and a nonempty boolean mask per image required")
    return (prediction-target).abs()[valid_mask.expand_as(prediction)].mean()


def evaluate_valid(model, prepared, input_mode, device):
    """Streaming, full-image independent valid only, selected RGB scalar aggregation."""
    model.eval()
    absolute, squared, elements = 0., 0., 0
    with torch.no_grad():
        for record in prepared["splits"]["valid"]:
            sample = _read_record(prepared, record)
            batch = _apply_mode(float_io.to_torch_bchw([sample], device=device), input_mode)
            del sample
            prediction = model(batch["input"])[-1]
            masked_spatial_l1(prediction, batch["target"], batch["valid_mask"])
            residual = (prediction-batch["target"])[batch["valid_mask"].expand_as(prediction)].double()
            absolute += residual.abs().sum().item()
            squared += residual.square().sum().item()
            elements += residual.numel()
    return {"mae": absolute/elements, "mse": squared/elements,
            "valid_rgb_elements": elements, "sample_count": len(prepared["splits"]["valid"])}


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_hash(model):
    return state_hash(model.state_dict())


def expected_adam_parameters(model, device):
    """Find the final-output dependency graph without touching RNG, buffers or .grad.

    The two low-resolution RGB heads have no direct final-only supervision;
    demanding Adam moments for every named parameter would be incorrect.
    eval() prevents dropout/BN state mutation; RNG is additionally forked.
    """
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    was_training = model.training
    devices = [device.index] if device.type == "cuda" else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            output = model(torch.zeros((1, 9, 8, 8), device=device))[-1]
            gradients = torch.autograd.grad(output.sum(), [p for _, p in named], allow_unused=True)
        return {name for (name, _), grad in zip(named, gradients) if grad is not None}
    finally:
        model.train(was_training)


def validate_adam_state(saved, model, completed_update, device):
    """Require every active parameter's actual moment tensors and completed step."""
    named = list(model.named_parameters())
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"} or not isinstance(saved["state"], dict):
        raise ValueError("Malformed Adam state")
    groups = saved["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1 or groups[0].get("params") != list(range(len(named))):
        raise ValueError("Adam parameter identity/order differs from the fixed model")
    active_names = expected_adam_parameters(model, device)
    expected = {i for i, (name, _) in enumerate(named) if name in active_names}
    if set(saved["state"]) != expected:
        raise ValueError("Missing/extra Adam state for final-output parameters")
    for index in expected:
        item, parameter = saved["state"][index], named[index][1]
        if not isinstance(item, dict) or set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Incomplete Adam moment/step fields")
        step = item["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all() or step.item() != completed_update:
            raise ValueError("Adam step differs from completed update")
        for key in ("exp_avg", "exp_avg_sq"):
            value = item[key]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                raise ValueError("Adam moment shape/dtype/value mismatch")
        if torch.any(item["exp_avg_sq"] < 0):
            raise ValueError("Adam second moment must be nonnegative")


def load_new_checkpoint(path, model, optimizer, expected_contract, device):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"model_state", "optimizer_state", "torch_rng_state", "cuda_rng_state",
                "history", "best_mae", "best_update", "best_model_state", "best_model_sha256",
                "model_sha256", "update"}
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA or checkpoint.get("contract") != expected_contract:
        raise ValueError("Not the exact v2 data/mode/code/budget contract; legacy weights rejected")
    update = checkpoint.get("update")
    if not required <= checkpoint.keys() or type(update) is not int or not 1 <= update < expected_contract["total_updates"] or not finite_tree(checkpoint):
        raise ValueError("Invalid/non-finite completed-update checkpoint")
    history = checkpoint["history"]
    if not isinstance(history, list) or len(history) != update or [r["update"] for r in history] != list(range(1, update+1)):
        raise ValueError("Checkpoint history is not complete and ordered")
    valid_rows = [r for r in history if "valid" in r]
    if not valid_rows or checkpoint["best_mae"] != min(r["valid"]["mae"] for r in valid_rows) or checkpoint["best_update"] != next(r["update"] for r in valid_rows if r["valid"]["mae"] == checkpoint["best_mae"]):
        raise ValueError("Checkpoint selection history inconsistent")
    best_row = next(r for r in valid_rows if r["update"] == checkpoint["best_update"])
    if state_hash(checkpoint["model_state"]) != checkpoint["model_sha256"] or state_hash(checkpoint["best_model_state"]) != checkpoint["best_model_sha256"] or best_row.get("model_sha256") != checkpoint["best_model_sha256"]:
        raise ValueError("Current/best model identity does not match selection history")
    if set(checkpoint["best_model_state"]) != set(model.state_dict()) or any(checkpoint["best_model_state"][k].shape != v.shape or checkpoint["best_model_state"][k].dtype != v.dtype for k, v in model.state_dict().items()):
        raise ValueError("Embedded best model is incompatible with the fixed architecture")
    validate_adam_state(checkpoint["optimizer_state"], model, update, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    specified = expected_contract["optimizer"]
    for group in optimizer.param_groups:
        if group["lr"] != specified["lr"] or list(group["betas"]) != specified["betas"] or group["eps"] != specified["eps"] or group["weight_decay"] != specified["weight_decay"]:
            raise ValueError("Optimizer settings differ")
        if any(group.get(key) is not specified[key] for key in ("foreach", "fused", "capturable", "amsgrad", "maximize")):
            raise ValueError("Optimizer execution policy differs")
    for parameter, state in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state:
                state[key] = state[key].to(parameter.device)
        # Adam step remains CPU for non-capturable/non-fused FP32 policy.
        if "step" in state:
            state["step"] = state["step"].cpu()
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if device.type == "cuda":
        if not isinstance(checkpoint["cuda_rng_state"], torch.Tensor):
            raise ValueError("CUDA run must carry its device RNG")
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
    elif checkpoint["cuda_rng_state"] is not None:
        raise ValueError("CPU contract cannot inherit CUDA RNG")
    return checkpoint


def run_training(allocation_path, output_directory=None, *, write=False, updates=1, num_res=20,
                 input_mode="full_maps", seed=0, learning_rate=1e-4, resume=None,
                 device="cpu", batch_size=1, patch_size=None, validation_interval=1,
                 checkpoint_interval=1000, stop_after=None, threads=1):
    integers = (updates, num_res, batch_size, validation_interval, checkpoint_interval, threads)
    if any(type(v) is not int or v < 1 for v in integers) or type(seed) is not int or not 0 <= seed < 2**32 or input_mode not in MODES or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid explicit training configuration")
    if patch_size is not None and (type(patch_size) is not int or patch_size < 4 or patch_size % 4):
        raise ValueError("Patch size must be divisible by 4")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1 or stop_after > updates):
        raise ValueError("stop_after must be within the frozen total update budget")
    if checkpoint_interval % validation_interval or (stop_after is not None and stop_after < updates and stop_after % validation_interval):
        raise ValueError("Checkpoint/early-stop boundaries must follow frozen validation points")
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA FP32 are supported")
    target_device = torch.device("cpu") if target_device.type == "cpu" else torch.device("cuda", target_device.index or 0)
    prepared = prepare_allocation(allocation_path)
    contract = {"float_schema": float_io.SCHEMA_VERSION, "input_mode": input_mode, "num_res": num_res,
                "architecture": "GAMDUNetPlus 9-channel from scratch", "seed": seed,
                "allocation_sha256": prepared["allocation_sha256"], "npz_records": prepared["npz_records"],
                "code_sha256": code_hashes(), "map_scale_px": prepared["map_scale_px"],
                "optimizer": {"name": "Adam", "lr": learning_rate, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.,
                              "foreach": False, "fused": False, "capturable": False, "amsgrad": False, "maximize": False},
                "loss": "final-resolution masked linear-RGB spatial L1 only; no clamp, FFT or multiscale loss",
                "validation": "full-image independent valid only; min unclipped linear-RGB scalar-weighted MAE",
                "sampling": "NumPy PCG64 SeedSequence([seed,update,0x1ADA]), with replacement; stateless crops; workers=0",
                "total_updates": updates, "batch_size": batch_size, "patch_size": patch_size,
                "validation_interval": validation_interval, "checkpoint_interval": checkpoint_interval,
                "checkpoint_saving": "full state at checkpoint_interval or explicit call-end; best cloned at every validation",
                "device": str(target_device), "precision": "float32", "threads": threads,
                "deterministic_algorithms": True, "TF32": False, "cudnn_benchmark": False}
    if not write:
        if resume is not None:
            raise ValueError("Resume requires explicit --write")
        return {"action": "streaming_preflight_only_no_training_or_writes", "contract": contract,
                "split_counts": {k: len(v) for k, v in prepared["splits"].items()},
                "maximum_sample_array_bytes": prepared["maximum_sample_array_bytes"]}
    if output_directory is None:
        raise ValueError("Explicit write requires a new output directory")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(output)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is not available; no silent CPU fallback")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if target_device.type == "cuda":
        with torch.cuda.device(target_device):
            torch.cuda.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats(target_device)
    model = build_plus(num_res, target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, fused=False, capturable=False)
    start, history, best_mae, best_update, best_state = 0, [], math.inf, 0, None
    if resume is not None:
        state = load_new_checkpoint(resume, model, optimizer, contract, target_device)
        start, history = state["update"], state["history"]
        best_mae, best_update, best_state = state["best_mae"], state["best_update"], state["best_model_state"]
    actual_end = min(updates, stop_after if stop_after is not None else updates)
    if actual_end <= start:
        raise ValueError("This call must advance the fixed-budget run")
    initial_hash, begun = parameter_hash(model), time.perf_counter()
    output.mkdir(exist_ok=False)
    (output/"allocation.json").write_bytes(Path(allocation_path).read_bytes())
    last_checkpoint_path = None
    for update in range(start+1, actual_end+1):
        model.train()
        plan = sample_plan(prepared, seed, update, batch_size, patch_size)
        batch = training_batch(prepared, plan, input_mode, target_device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["input"])[-1]
        loss = masked_spatial_l1(prediction, batch["target"], batch["valid_mask"])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        row = {"update": update, "train_masked_l1": loss.item(), "gradient_l2_norm": norm.item(),
               "sampling_plan_sha256": hashlib.sha256(json.dumps(plan, separators=(",", ":")).encode()).hexdigest()}
        validate_now = update % validation_interval == 0 or update == actual_end
        checkpoint_path = output/f"update-{update:08d}.pt"
        if validate_now:
            row["valid"] = evaluate_valid(model, prepared, input_mode, target_device)
            row["model_sha256"] = parameter_hash(model)
            if row["valid"]["mae"] < best_mae:
                best_mae, best_update = row["valid"]["mae"], update
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(row)
        if update % checkpoint_interval == 0 or update == actual_end:
            # A checkpoint must include a selection observation; validation above
            # always runs at first checkpoint if interval exceeds the budget.
            if best_state is None:
                raise RuntimeError("Checkpoint before a frozen validation point is forbidden")
            checkpoint = {"checkpoint_schema": CHECKPOINT_SCHEMA, "contract": contract, "update": update,
                          "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                          "torch_rng_state": torch.get_rng_state(),
                          "cuda_rng_state": torch.cuda.get_rng_state(target_device) if target_device.type == "cuda" else None,
                          "history": history, "best_mae": best_mae, "best_update": best_update,
                          "model_sha256": parameter_hash(model), "best_model_state": best_state,
                          "best_model_sha256": state_hash(best_state)}
            if not finite_tree(checkpoint):
                raise ValueError("Non-finite model, optimizer or history; no valid checkpoint")
            with checkpoint_path.open("xb") as destination:
                torch.save(checkpoint, destination)
            last_checkpoint_path = checkpoint_path
        print(json.dumps({"event": "training_progress", "input_mode": input_mode, **row}), flush=True)
    report = {"status": "new v2 float-contract optimization; no historical or industrial performance certification",
              "contract": contract, "completed_update": actual_end, "requested_total_updates": updates,
              "complete_budget": actual_end == updates, "resumed_from": str(resume) if resume else None,
              "history": history, "best_update": best_update, "best_valid_mae": best_mae,
              "best_model": {"checkpoint": str(last_checkpoint_path.resolve()), "state_key": "best_model_state",
                             "sha256": state_hash(best_state), "selected_update": best_update},
              "initial_parameter_sha256": initial_hash, "final_parameter_sha256": parameter_hash(model),
              "checkpoint_files": {p.name: file_hash(p) for p in output.glob("update-*.pt")},
              "wall_seconds_including_validation_checkpointing": time.perf_counter()-begun,
              "maximum_preflight_sample_array_bytes": prepared["maximum_sample_array_bytes"],
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                              "device": str(target_device), "cuda_executed": target_device.type == "cuda",
                              "CUDA": torch.version.cuda, "threads": threads,
                              "GPU": torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else None,
                              "CUDA_peak_allocated_bytes": torch.cuda.max_memory_allocated(target_device) if target_device.type == "cuda" else None},
              "test_split": "file/hash/contract preflight only; never forwarded, optimized or used for selection"}
    with (output/"run.json").open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", required=True, type=Path)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--num-res", type=int, default=20)
    parser.add_argument("--input-mode", choices=MODES, default="full_maps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    report = run_training(args.allocation, args.write, write=args.write is not None,
                          updates=args.updates, num_res=args.num_res, input_mode=args.input_mode,
                          seed=args.seed, learning_rate=args.learning_rate, resume=args.resume,
                          device=args.device, batch_size=args.batch_size, patch_size=args.patch_size,
                          validation_interval=args.validation_interval, checkpoint_interval=args.checkpoint_interval,
                          stop_after=args.stop_after, threads=args.threads)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
