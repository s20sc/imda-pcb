"""Strict NEW float32 xy0 NPZ interface; never a JPEG/legacy decoder.

No training or industrial validation; legacy PSNR/SSIM cannot be inherited.
NumPy is the only eager dependency. Torch is imported only by to_torch_bchw.
All network arrays are linear RGB / signed displacement float32, layout HWC9:
[R,G,B,Cdx,Cdy,0,Edx,Edy,0]. Map components are pixels / map_scale_px.
The only file-writing API is explicit write_npz_sample(), which refuses overwrite.
Unversioned imu_synthesis.main NPZ output must be explicitly wrapped with actual
schema/provenance via this writer; no split IDs or provenance are invented.
"""
import copy
import json
from pathlib import Path
import re
import zipfile

import numpy as np

SCHEMA_VERSION = "imda-float-xy0-v2"
CHANNEL_ORDER = ["R_linear", "G_linear", "B_linear", "C_dx", "C_dy", "0", "E_dx", "E_dy", "0"]
REQUIRED_STRING_PROVENANCE_FIELDS = (
    "sample_id", "source_dataset_id", "source_dataset_version", "source_image_id",
    "group_kind", "group_id", "trajectory_family_id", "trajectory_id",
    "original_split_id", "split_name", "split_manifest_sha256")
NULLABLE_PROVENANCE_FIELDS = ("physical_scene_id", "physical_session_id")
PROVENANCE_FIELDS = REQUIRED_STRING_PROVENANCE_FIELDS + NULLABLE_PROVENANCE_FIELDS
GROUP_KINDS = ("source_image", "verified_duplicate_family", "physical_scene")
COLOR_SOURCE_FIELDS = ("input_encoding", "linearization", "profile_origin", "icc_profile_sha256")
SOURCE_METADATA_FIELDS = ("camera_parameter_origin", "color_source")
REQUIRED_METADATA_FIELDS = (
    "schema_version", "layout", "photometric_space", "network_channel_order",
    "legacy_compatible", "map_scale_px", "provenance") + SOURCE_METADATA_FIELDS
OPTIONAL_METADATA_FIELDS = (
    "implementation", "empirical_status", "geometry", "units", "descriptor_encoding",
    "error_basis", "loader_requirement", "boundary", "mask_scope", "error_summary",
    "exposure_weight_sum", "input_source", "augmentation_history")
_PLACEHOLDERS = frozenset(("unknown", "null", "none", "n/a", "na", "unspecified", "not_available"))
REQUIRED_ARRAYS = ("blur_linear_rgb", "cp_map", "ep_map", "network_input_9", "exposure_coverage",
                   "full_exposure_valid", "fit_valid")
OPTIONAL_ARRAYS = ("sharp_linear_rgb", "control_px", "endpoint_px", "rms_error_px", "max_error_px",
                   "trajectory_valid", "sample_times_seconds", "exposure_weights", "fit_times_seconds")


def _exact_fields(value, required, optional=(), name="object"):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} field names must be strings")
    missing = set(required) - set(value)
    extra = set(value) - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{name} fields mismatch: missing={sorted(missing)} extra={sorted(extra)}")


def _identifier(value, name):
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or value.casefold() in _PLACEHOLDERS):
        raise ValueError(f"{name} must be a nonempty non-placeholder string without outer whitespace")


def _sha256(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")


def validate_provenance(provenance, expected_split=None, expected_manifest_sha256=None):
    """Check one declared source record, not the truth of IDs or cross-split independence.

    Physical scene/session IDs unknown to the caller are required JSON null.
    Source-image grouping is honest image disjointness, not physical independence.
    The allocation digest is supplied by the caller; no allocation is invented.
    """
    _exact_fields(provenance, PROVENANCE_FIELDS, name="provenance")
    for key in REQUIRED_STRING_PROVENANCE_FIELDS:
        _identifier(provenance[key], key)
    for key in NULLABLE_PROVENANCE_FIELDS:
        if provenance[key] is not None:
            _identifier(provenance[key], key)
    kind = provenance["group_kind"]
    if kind not in GROUP_KINDS:
        raise ValueError(f"group_kind must be one of {GROUP_KINDS}")
    if kind == "source_image" and provenance["group_id"] != provenance["source_image_id"]:
        raise ValueError("source_image group_id must equal source_image_id")
    if kind == "physical_scene" and (provenance["physical_scene_id"] is None
                                     or provenance["group_id"] != provenance["physical_scene_id"]):
        raise ValueError("physical_scene grouping requires a known physical_scene_id equal to group_id")
    _sha256(provenance["split_manifest_sha256"], "split_manifest_sha256")
    if expected_split is not None:
        _identifier(expected_split, "expected_split")
        if provenance["split_name"] != expected_split:
            raise ValueError("Sample split_name does not match requested split")
    if expected_manifest_sha256 is not None:
        _sha256(expected_manifest_sha256, "expected_manifest_sha256")
        if provenance["split_manifest_sha256"] != expected_manifest_sha256:
            raise ValueError("Sample split_manifest_sha256 does not match expected allocation manifest")
    return provenance


def validate_source_metadata(metadata):
    """Require an explicit camera origin and color conversion claim; do not certify calibration."""
    if metadata.get("camera_parameter_origin") not in ("simulated", "calibrated"):
        raise ValueError("camera_parameter_origin must explicitly be simulated or calibrated")
    color = metadata.get("color_source")
    _exact_fields(color, COLOR_SOURCE_FIELDS, name="color_source")
    encoding, transform, origin, icc = (color[key] for key in COLOR_SOURCE_FIELDS)
    if encoding == "linear_rgb":
        if (transform, origin, icc) != ("none", "declared_linear", None):
            raise ValueError("linear_rgb source requires none/declared_linear/null ICC")
    elif encoding == "srgb":
        if transform != "inverse_srgb":
            raise ValueError("srgb source requires explicit inverse_srgb linearization")
        if origin == "assumed_srgb":
            if icc is not None:
                raise ValueError("assumed_srgb requires null ICC digest")
        elif origin == "icc_converted_srgb":
            _sha256(icc, "icc_profile_sha256")
        else:
            raise ValueError("srgb profile_origin must be assumed_srgb or icc_converted_srgb")
    else:
        raise ValueError("color_source input_encoding must be srgb or linear_rgb")
    return metadata


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON metadata constant: {value}")


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON metadata field: {key}")
        value[key] = item
    return value


def _rgb(array, name, shape):
    if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float32 with shape {shape}")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} must be linear RGB in [0,1]; no automatic gamma decoding")


def validate_sample(sample, expected_split=None, require_target=False, *, expected_manifest_sha256=None):
    """Validate metadata and numerical interface; numerical range cannot prove color space.

    fit_valid is a forward descriptor mask, full_exposure_valid is an inverse
    exposure mask. valid_mask intersects availability on the shared pixel grid;
    it is an explicit loader policy, not a proof of physical correspondence.
    """
    if not isinstance(sample, dict) or not isinstance(sample.get("metadata"), dict):
        raise ValueError("A sample dictionary with metadata is required")
    metadata = sample["metadata"]
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unversioned/incompatible NPZ; v1 has no implicit migration to imda-float-xy0-v2")
    _exact_fields(sample, REQUIRED_ARRAYS + ("metadata",), OPTIONAL_ARRAYS + ("valid_mask",), "sample")
    _exact_fields(metadata, REQUIRED_METADATA_FIELDS, OPTIONAL_METADATA_FIELDS, "metadata")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Metadata must contain only finite JSON values") from error
    validate_source_metadata(metadata)
    if "augmentation_history" in metadata:
        if not isinstance(metadata["augmentation_history"], list):
            raise ValueError("augmentation_history must be a list")
        for operation in metadata["augmentation_history"]:
            _identifier(operation, "augmentation_history entry")
    if metadata.get("photometric_space") != "linear_rgb" or metadata.get("layout") != "HWC9":
        raise ValueError("Only declared linear_rgb/HWC9 is supported")
    if metadata.get("network_channel_order") != CHANNEL_ORDER or metadata.get("legacy_compatible") is not False:
        raise ValueError("New xy0 channel order must be explicit; legacy compatibility is not assumed")
    scale = metadata.get("map_scale_px")
    limits = np.finfo(np.float32)
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not np.isfinite(scale) or not float(limits.tiny) <= scale <= float(limits.max):
        raise ValueError("map_scale_px must lie in float32 positive normal range")
    validate_provenance(metadata["provenance"], expected_split, expected_manifest_sha256)
    for key in REQUIRED_ARRAYS:
        if key not in sample or not isinstance(sample[key], np.ndarray):
            raise ValueError(f"Missing array: {key}")
    for key in REQUIRED_ARRAYS + OPTIONAL_ARRAYS:
        if key in sample and (not isinstance(sample[key], np.ndarray) or sample[key].dtype.kind not in "biuf"):
            raise ValueError(f"{key} must be a real numeric/bool NumPy array, never object/pickle")
    blur = sample["blur_linear_rgb"]
    if blur.ndim != 3 or blur.shape[-1] != 3 or min(blur.shape[:2]) < 1:
        raise ValueError("blur_linear_rgb must be nonempty H,W,3")
    height, width = blur.shape[:2]
    shape = (height, width, 3)
    _rgb(blur, "blur_linear_rgb", shape)
    if require_target and "sharp_linear_rgb" not in sample:
        raise ValueError("Supervised use requires explicit sharp_linear_rgb; no target is synthesized by this loader")
    if "sharp_linear_rgb" in sample:
        _rgb(sample["sharp_linear_rgb"], "sharp_linear_rgb", shape)
    for name in ("cp_map", "ep_map"):
        value = sample[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite H,W,3 float32")
        if np.any(value[..., 2] != 0):
            raise ValueError(f"{name} channel 2 must be exactly zero for xy0")
    network = sample["network_input_9"]
    if network.dtype != np.float32 or network.shape != (height, width, 9) or not np.isfinite(network).all():
        raise ValueError("network_input_9 must be finite H,W,9 float32")
    if not np.array_equal(network, np.concatenate([blur, sample["cp_map"], sample["ep_map"]], axis=-1)):
        raise ValueError("Stored network_input_9 disagrees with RGB/C/E component arrays")
    coverage = sample["exposure_coverage"]
    if coverage.shape != (height, width) or coverage.dtype != np.float32 or not np.isfinite(coverage).all() or np.any(coverage < 0) or np.any(coverage > 1):
        raise ValueError("exposure_coverage must be finite H,W float32 in [0,1]")
    for name in ("full_exposure_valid", "fit_valid"):
        if sample[name].shape != (height, width) or sample[name].dtype != np.bool_:
            raise ValueError(f"{name} must be H,W bool")
    if not np.array_equal(sample["full_exposure_valid"], np.isclose(coverage, 1, atol=1e-7, rtol=0)):
        raise ValueError("full_exposure_valid disagrees with weighted coverage")
    valid = sample["fit_valid"]
    for name in ("cp_map", "ep_map"):
        if np.any(sample[name][~valid] != 0):
            raise ValueError("Descriptors with invalid fit must be zero-filled and accompanied by mask")
    for key, map_key in (("control_px", "cp_map"), ("endpoint_px", "ep_map")):
        if key in sample:
            value = sample[key]
            if value.shape != (height, width, 2) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite H,W,2 float32")
            decoded = sample[map_key][..., :2].astype(np.float64)*scale
            original = value.astype(np.float64)
            nonzero = original != 0
            if not np.isfinite(decoded).all() or np.any(decoded[nonzero] == 0) or np.any(np.abs(decoded[nonzero]-original[nonzero]) > 1e-6*np.abs(original[nonzero])) or np.any(decoded[~nonzero] != 0):
                raise ValueError(f"{key} disagrees with normalized descriptor")
    for name in ("rms_error_px", "max_error_px"):
        if name in sample:
            error = sample[name]
            if error.shape != (height, width) or error.dtype != np.float32:
                raise ValueError(f"{name} must be H,W float32")
            if not np.isfinite(error[valid]).all() or np.any(error[valid] < 0) or not np.isnan(error[~valid]).all():
                raise ValueError(f"{name}: nonnegative finite on valid fits; NaN only/exactly on invalid fits")
    if "trajectory_valid" in sample:
        mask = sample["trajectory_valid"]
        if mask.ndim != 3 or mask.shape[1:] != (height, width) or mask.dtype != np.bool_ or not np.array_equal(mask.all(axis=0), valid):
            raise ValueError("trajectory_valid must be T,H,W bool agreeing with fit_valid")
    if "exposure_weights" in sample or "sample_times_seconds" in sample:
        if "exposure_weights" not in sample or "sample_times_seconds" not in sample:
            raise ValueError("Exposure samples and weights must be supplied together")
        weights, times = sample["exposure_weights"], sample["sample_times_seconds"]
        if weights.ndim != 1 or times.shape != weights.shape or len(times) == 0 or not np.isfinite(times).all() or not np.isfinite(weights).all() or np.any(weights <= 0) or np.any(np.diff(times) <= 0) or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
            raise ValueError("Exposure samples must increase with positive normalized weights")
    if "fit_times_seconds" in sample:
        times = sample["fit_times_seconds"]
        if times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("At least three increasing finite fit times are required")
        if "trajectory_valid" in sample and sample["trajectory_valid"].shape[0] != len(times):
            raise ValueError("fit_times_seconds and trajectory_valid length mismatch")
    sample["valid_mask"] = sample["fit_valid"] & sample["full_exposure_valid"]
    return sample


def write_npz_sample(path, synthesis_result, provenance, sharp_linear_rgb=None, *, source_metadata=None):
    """Explicitly write a new schema-versioned lossless file; never infer split IDs.

    provenance must identify the actual frozen source/split manifest, not a hash
    manufactured by this function. This persists claims, not their truth proof.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Only .npz is supported")
    _exact_fields(synthesis_result, REQUIRED_ARRAYS + ("metadata",), OPTIONAL_ARRAYS + ("valid_mask",), "synthesis_result")
    metadata = copy.deepcopy(synthesis_result["metadata"])
    if not isinstance(metadata, dict):
        raise ValueError("Synthesis metadata must be an object")
    if metadata.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("Writer refuses implicit migration of an old/incompatible schema")
    if "provenance" in metadata and metadata["provenance"] != provenance:
        raise ValueError("Writer refuses changing existing provenance")
    for key, expected in (("layout", "HWC9"), ("photometric_space", "linear_rgb")):
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"Writer refuses overriding incompatible {key}")
    if source_metadata is not None:
        _exact_fields(source_metadata, SOURCE_METADATA_FIELDS, name="source_metadata")
        for key in SOURCE_METADATA_FIELDS:
            if key in metadata and metadata[key] != source_metadata[key]:
                raise ValueError(f"Conflicting source metadata: {key}")
            metadata[key] = copy.deepcopy(source_metadata[key])
    metadata.update(schema_version=SCHEMA_VERSION, layout="HWC9", photometric_space="linear_rgb",
                    provenance=copy.deepcopy(provenance))
    sample = {key: np.asarray(synthesis_result[key]) for key in REQUIRED_ARRAYS + OPTIONAL_ARRAYS if key in synthesis_result}
    sample["metadata"] = metadata
    if sharp_linear_rgb is not None:
        sample["sharp_linear_rgb"] = np.asarray(sharp_linear_rgb)
    validate_sample(sample)
    serialized_metadata = np.array(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    # No implicit mkdir or replacement; existing files fail with FileExistsError.
    with path.open("xb") as output:
        np.savez_compressed(output, **{key: value for key, value in sample.items() if key not in ("metadata", "valid_mask")},
                            metadata_json=serialized_metadata)


def load_npz_sample(path, expected_split=None, require_target=False, *, expected_manifest_sha256=None):
    """Load only the new schema, with allow_pickle=False and strict validation."""
    path = Path(path)
    allowed = set(REQUIRED_ARRAYS + OPTIONAL_ARRAYS + ("metadata_json",))
    try:
        with zipfile.ZipFile(path) as zipped:
            names = zipped.namelist()
            if len(names) != len(set(names)) or any(name not in {key+".npy" for key in allowed} for name in names):
                raise ValueError("NPZ contains duplicate or unsupported member names")
    except zipfile.BadZipFile as error:
        raise ValueError("Only a safe NPZ archive is supported") from error
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("Missing metadata_json; legacy/JPEG inputs are not supported")
        scalar = archive["metadata_json"]
        if scalar.shape != () or scalar.dtype.kind != "U":
            raise ValueError("metadata_json must be a Unicode scalar, not a pickle/object")
        metadata = json.loads(str(scalar), parse_constant=_reject_constant, object_pairs_hook=_unique_json_object)
        sample = {key: archive[key].copy() for key in REQUIRED_ARRAYS + OPTIONAL_ARRAYS if key in archive.files}
        sample["metadata"] = metadata
    return validate_sample(sample, expected_split, require_target, expected_manifest_sha256=expected_manifest_sha256)


def horizontal_flip(sample):
    """NEW xy0 semantics: D'(F(p))=F(p+D(p))-F(p)=(-dx,dy).

    Flip image positions, target and masks too. Does not infer JPEG/BGR semantics.
    """
    validate_sample(sample)
    result = copy.deepcopy(sample)
    for key in ("blur_linear_rgb", "cp_map", "ep_map", "sharp_linear_rgb", "exposure_coverage",
                "full_exposure_valid", "fit_valid", "rms_error_px", "max_error_px", "control_px", "endpoint_px"):
        if key in result:
            result[key] = sample[key][:, ::-1].copy(order="C")
    for key in ("cp_map", "ep_map", "control_px", "endpoint_px"):
        if key in result:
            result[key][..., 0] *= -1
    if "trajectory_valid" in result:
        result["trajectory_valid"] = sample["trajectory_valid"][:, :, ::-1].copy(order="C")
    result["network_input_9"] = np.concatenate([result["blur_linear_rgb"], result["cp_map"], result["ep_map"]], axis=-1)
    result["metadata"].setdefault("augmentation_history", []).append("horizontal_flip_xy0")
    return validate_sample(result)


def collate_bchw(samples):
    """Validate and stack equal-resolution samples to contiguous NumPy BCHW."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    samples = [validate_sample(sample) for sample in samples]
    shape = samples[0]["network_input_9"].shape
    scale = samples[0]["metadata"]["map_scale_px"]
    if any(sample["network_input_9"].shape != shape or sample["metadata"]["map_scale_px"] != scale for sample in samples):
        raise ValueError("A batch requires equal spatial shape and map_scale_px")
    targets = ["sharp_linear_rgb" in sample for sample in samples]
    if any(targets) and not all(targets):
        raise ValueError("Do not mix supervised and target-free samples in a batch")
    result = {"input": np.ascontiguousarray(np.stack([s["network_input_9"] for s in samples]).transpose(0, 3, 1, 2)),
              "valid_mask": np.stack([s["valid_mask"] for s in samples])[:, None],
              "metadata": [copy.deepcopy(s["metadata"]) for s in samples]}
    if all(targets):
        result["target"] = np.ascontiguousarray(np.stack([s["sharp_linear_rgb"] for s in samples]).transpose(0, 3, 1, 2))
    return result


def to_torch_bchw(samples, device=None):
    """Lazy Torch adapter; no model, optimizer, training, or checkpoint loading."""
    import torch
    batch = collate_bchw(samples)
    result = {key: torch.from_numpy(value).to(device=device) if isinstance(value, np.ndarray) else value
              for key, value in batch.items()}
    return result


class FloatNPZDataset:
    """Map-style NumPy dataset usable by a caller-owned Torch DataLoader.

    No shuffle, splitting, crop, augmentation randomness, or training is hidden.
    Use collate_bchw/to_torch_bchw explicitly; only metadata supplied by the
    caller establishes split identity, and it does not certify leakage absence.
    """
    def __init__(self, paths, expected_split=None, require_target=False, transform=None, *, expected_manifest_sha256=None):
        self.paths = tuple(Path(path) for path in paths)
        if len(set(path.resolve() for path in self.paths)) != len(self.paths):
            raise ValueError("Duplicate NPZ path in dataset")
        self.expected_split, self.require_target, self.transform = expected_split, require_target, transform
        self.expected_manifest_sha256 = expected_manifest_sha256

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sample = load_npz_sample(self.paths[index], self.expected_split, self.require_target,
                                 expected_manifest_sha256=self.expected_manifest_sha256)
        if self.transform is not None:
            sample = self.transform(sample)
        return validate_sample(sample, self.expected_split, self.require_target,
                               expected_manifest_sha256=self.expected_manifest_sha256)
