"""Explicit RGB/ICC, simulated camera, common ROI and original-label geometry.

Pure array/PIL helpers; no file writes, data selection, training or inference.
Resizing uses ONE isotropic pixel-center scale. The short side is trimmed by
less than four target pixels to obtain a multiple of four, never stretched.
Masks remain on their original reference grid; scores are mapped back to it.
"""
import copy
import hashlib
import importlib.util
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms
from scipy.ndimage import map_coordinates, gaussian_filter


def _renderer():
    spec = importlib.util.spec_from_file_location("industrial_geometry_renderer", Path(__file__).with_name("imu_synthesis.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_unit_rgb(value, *, clip=False):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1 or not np.isfinite(value).all():
        raise ValueError("Expected finite nonempty HWC RGB")
    if clip:
        return np.clip(value, 0, 1)
    if value.min() < 0 or value.max() > 1:
        raise ValueError("RGB must be in [0,1]; clipping requires an explicit request")
    return value


def srgb_to_linear(srgb):
    """Decode standard sRGB once; reject nonfinite/out-of-range source values."""
    value = _finite_unit_rgb(srgb)
    return np.where(value <= .04045, value/12.92, ((value+.055)/1.055)**2.4).astype(np.float32)


def linear_to_srgb(linear, *, clip=False):
    """Encode once for display/detection. Default rejects; clip=True clips first.

    Never use this encoder to clamp a training loss prediction implicitly.
    """
    value = _finite_unit_rgb(linear, clip=clip)
    return np.where(value <= .0031308, 12.92*value, 1.055*value**(1/2.4)-.055).astype(np.float32)


def decode_rgb(image):
    """Decode a caller-opened RGB PIL image to float32 linear RGB.

    ICC present: actual LittleCMS conversion to sRGB, relative-colorimetric,
    flags=0, then inverse sRGB. No ICC: explicitly assume standard sRGB.
    Gray/CMYK/alpha/palette and nontrivial EXIF orientation fail, never coerce.
    """
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("Only explicit PIL RGB is accepted; no gray/CMYK/alpha conversion")
    if image.getexif().get(274, 1) != 1:
        raise ValueError("EXIF orientation must be resolved jointly with labels before this adapter")
    original = image.copy()
    icc = image.info.get("icc_profile")
    if icc is None:
        converted = original
        source = {"input_encoding": "srgb", "linearization": "inverse_srgb",
                  "profile_origin": "assumed_srgb", "icc_profile_sha256": None}
    else:
        if not isinstance(icc, bytes) or not icc:
            raise ValueError("ICC profile must be nonempty bytes; cannot assume sRGB after a profile error")
        try:
            profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            converted = ImageCms.profileToProfile(original, profile, ImageCms.createProfile("sRGB"),
                                                  renderingIntent=1, outputMode="RGB", flags=0)
        except Exception as error:
            raise ValueError("ICC to sRGB conversion failed; no fallback to assumed sRGB") from error
        source = {"input_encoding": "srgb", "linearization": "inverse_srgb",
                  "profile_origin": "icc_converted_srgb", "icc_profile_sha256": hashlib.sha256(icc).hexdigest()}
    linear = srgb_to_linear(np.asarray(converted, dtype=np.float64)/255.)
    return linear, source


def resize_geometry(original_size_wh, long_side):
    """Return one scale and source->full pixel-center affine, plus trimmed size."""
    if (len(original_size_wh) != 2 or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1 for v in original_size_wh)
            or not isinstance(long_side, int) or isinstance(long_side, bool) or long_side < 4 or long_side % 4):
        raise ValueError("Positive integer source size and long_side>=4 divisible by four required")
    width, height = map(int, original_size_wh)
    scale = long_side/max(width, height)
    full_w, full_h = (4*math.floor(width*scale/4+1e-12), 4*math.floor(height*scale/4+1e-12))
    if min(full_w, full_h) < 4:
        raise ValueError("Aspect ratio leaves no four-pixel short side")
    affine = np.array([[scale, 0, (scale-1)/2], [0, scale, (scale-1)/2], [0, 0, 1]], dtype=np.float64)
    return {"original_size_wh": [width, height], "full_size_wh": [full_w, full_h],
            "isotropic_scale": scale, "original_to_full_pixel_centers": affine.tolist(),
            "short_side_rounding": "isotropic center sampling; trim right/bottom fractional extent to multiple of four",
            "resampling": "linear RGB spatial Gaussian then bilinear order1; nearest edge extension",
            "antialias": {"sigma_original_pixels": max(0., (1/scale-1)/2),
                          "truncate": 4., "mode": "nearest", "channels_mixed": False}}


def resize_linear(linear, mapping):
    value = _finite_unit_rgb(linear)
    if [value.shape[1], value.shape[0]] != mapping["original_size_wh"]:
        raise ValueError("Mapping source dimensions disagree")
    width, height = mapping["full_size_wh"]
    scale = mapping["isotropic_scale"]
    sigma = mapping["antialias"]["sigma_original_pixels"]
    if sigma > 0:
        value = gaussian_filter(value, sigma=(sigma, sigma, 0), mode="nearest", truncate=4.)
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    coordinates = [(yy+.5)/scale-.5, (xx+.5)/scale-.5]
    return np.stack([map_coordinates(value[..., channel], coordinates, order=1, mode="nearest", prefilter=False)
                     for channel in range(3)], axis=-1).astype(np.float32)


def simulated_k(size_wh, horizontal_fov_degrees):
    """Explicit virtual square-pixel pinhole K, centers at (W-1)/2,(H-1)/2."""
    width, height = size_wh
    if min(width, height) < 1 or not np.isfinite(horizontal_fov_degrees) or not 0 < horizontal_fov_degrees < 180:
        raise ValueError("Positive size and horizontal FOV in (0,180) required")
    focal = width/(2*math.tan(math.radians(horizontal_fov_degrees)/2))
    return np.array([[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1]], dtype=np.float64)


def roi_geometry(mapping, margin):
    mapping = copy.deepcopy(mapping)
    width, height = mapping["full_size_wh"]
    if not isinstance(margin, int) or isinstance(margin, bool) or margin < 0:
        raise ValueError("margin must be a nonnegative integer")
    crop_w, crop_h = width-2*margin, height-2*margin
    if min(crop_w, crop_h) < 4 or crop_w % 4 or crop_h % 4:
        raise ValueError("Common cropped ROI must be nonempty and divisible by four")
    transform = np.array([[1, 0, -margin], [0, 1, -margin], [0, 0, 1]], dtype=np.float64)
    affine = transform @ np.asarray(mapping["original_to_full_pixel_centers"])
    scale = mapping["isotropic_scale"]
    original_w, original_h = mapping["original_size_wh"]
    # Original pixel CENTERS inside the cropped target pixel-edge footprint.
    x0 = max(0, math.ceil(margin/scale-.5))
    y0 = max(0, math.ceil(margin/scale-.5))
    x1 = min(original_w, math.ceil((margin+crop_w)/scale-.5))
    y1 = min(original_h, math.ceil((margin+crop_h)/scale-.5))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI contains no original reference pixel centers")
    mapping.update(margin=margin, crop_box_full_xyxy=[margin, margin, width-margin, height-margin],
                   crop_size_wh=[crop_w, crop_h], original_to_crop_pixel_centers=affine.tolist(),
                   original_roi_index_xyxy=[x0, y0, x1, y1],
                   roi_membership="original integer pixel centers within crop pixel-edge footprint [-.5,W-.5)x[-.5,H-.5)",
                   score_inverse_mapping="bilinear order1 at affine(original centers); nearest edge extension within ROI only")
    return mapping


def crop_synthesis(result, margin):
    """Crop EVERY known spatial array; vectors retain full-canvas pixel units."""
    height, width = result["blur_linear_rgb"].shape[:2]
    if width % 4 or height % 4:
        raise ValueError("Full synthesis canvas must be divisible by four before cropping")
    dummy = resize_geometry((width, height), max(width, height))
    roi_geometry(dummy, margin)  # validate a common /4 ROI before slicing
    ys, xs = slice(margin, height-margin), slice(margin, width-margin)
    cropped = {}
    for key, value in result.items():
        if key == "metadata":
            cropped[key] = copy.deepcopy(value)
        elif isinstance(value, np.ndarray):
            if key == "trajectory_valid":
                if value.ndim != 3 or value.shape[1:] != (height, width):
                    raise ValueError("trajectory_valid must have T,H,W layout")
                cropped[key] = value[:, ys, xs].copy()
            elif value.ndim >= 2 and value.shape[:2] == (height, width):
                cropped[key] = value[ys, xs].copy()
            elif value.ndim >= 3 and value.shape[1:3] == (height, width):
                cropped[key] = value[:, ys, xs].copy()
            elif value.ndim == 1:
                cropped[key] = value.copy()
            else:
                raise ValueError("Unknown spatial/debug array layout: "+key)
        else:
            raise ValueError("Unexpected synthesis field: "+key)
    if not cropped["full_exposure_valid"].all() or not cropped["fit_valid"].all():
        raise ValueError("Predetermined ROI includes invalid exposure/fit; reject, do not adapt ROI to outcomes")
    cropped["metadata"]["error_summary"] = _renderer().error_summary(cropped)
    cropped["metadata"]["error_summary"]["scope"] = "cropped common ROI; errors at supplied fit times only"
    return cropped


def original_annotation(mask, image_label, mapping):
    """Preserve original class IDs and caller's image label; never downsample GT."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype.kind not in "biu" or np.any(mask < 0):
        raise ValueError("Original defect mask must be nonnegative integer H,W; zero is normal")
    if [mask.shape[1], mask.shape[0]] != mapping["original_size_wh"]:
        raise ValueError("Original mask/image dimensions must match")
    if not isinstance(image_label, (str, int, bool)):
        raise ValueError("image_label must be explicitly supplied as a JSON scalar, never inferred")
    x0, y0, x1, y1 = mapping["original_roi_index_xyxy"]
    roi = mask[y0:y1, x0:x1].copy()
    classes = {}
    for label in np.unique(mask):
        if label == 0:
            continue
        full, retained = int(np.count_nonzero(mask == label)), int(np.count_nonzero(roi == label))
        classes[str(int(label))] = {"original_pixels": full, "retained_pixels": retained,
                                    "lost_pixels": full-retained, "completely_lost": retained == 0}
    stats = {"image_label": image_label, "original_defect_pixels": int(np.count_nonzero(mask)),
             "retained_defect_pixels": int(np.count_nonzero(roi)),
             "all_defects_outside_roi": bool(np.any(mask) and not np.any(roi)),
             "completely_lost_class_count": sum(v["completely_lost"] for v in classes.values()),
             "classes": classes, "statistics_unit": "per class and per image; not connected defect instances"}
    return {"original_mask": mask.copy(), "roi_mask_original_grid": roi, "statistics": stats}


def remap_score_to_original_roi(score_crop, mapping):
    """Map 2D score to original reference ROI pixel centers; no GT resampling."""
    score = np.asarray(score_crop, dtype=np.float64)
    if score.ndim != 2 or list(score.shape[::-1]) != mapping["crop_size_wh"] or not np.isfinite(score).all():
        raise ValueError("Score must be finite and match the cropped synthesis grid")
    x0, y0, x1, y1 = mapping["original_roi_index_xyxy"]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    affine = np.asarray(mapping["original_to_crop_pixel_centers"])
    u = affine[0, 0]*xx+affine[0, 2]
    v = affine[1, 1]*yy+affine[1, 2]
    return map_coordinates(score, [v, u], order=1, mode="nearest", prefilter=False)


def prepare_industrial_sample(image, mask_original, image_label, *, long_side, margin,
                              times_seconds, angular_rates_rad_s, k_original=None,
                              horizontal_fov_degrees=None, samples_per_interval=2, map_scale_px=1.):
    """Return (raw synthesis result, linear sharp target, original-grid annotation).

    Exactly one K choice is required. Caller-provided K is ORIGINAL-grid virtual
    K, transformed with pixel-center resize. FOV defines virtual FULL-canvas K.
    No calibration is invented. Raw result is passed to explicit v2 NPZ writer
    with caller-owned provenance; source mapping is nested in input_source.
    """
    if (k_original is None) == (horizontal_fov_degrees is None):
        raise ValueError("Provide exactly one simulated k_original or horizontal_fov_degrees")
    renderer = _renderer()
    linear, color = decode_rgb(image)
    mapping = roi_geometry(resize_geometry(image.size, long_side), margin)
    full = resize_linear(linear, mapping)
    if k_original is not None:
        k_full = np.asarray(mapping["original_to_full_pixel_centers"]) @ renderer.validate_k(k_original)
        mapping["camera_definition"] = "caller-supplied simulated original-grid K, pixel-center scaled"
    else:
        k_full = simulated_k(mapping["full_size_wh"], horizontal_fov_degrees)
        mapping["camera_definition"] = {"simulated_full_canvas_horizontal_fov_degrees": float(horizontal_fov_degrees)}
    translate = np.array([[1, 0, -margin], [0, 1, -margin], [0, 0, 1]])
    mapping.update(k_full=k_full.tolist(), k_crop=(translate@k_full).tolist())
    full_result = renderer.synthesize(full, k_full, times_seconds, angular_rates_rad_s,
                                      samples_per_interval=samples_per_interval, map_scale_px=map_scale_px,
                                      camera_parameter_origin="simulated", color_source=color)
    result = crop_synthesis(full_result, margin)
    x0, y0, x1, y1 = mapping["crop_box_full_xyxy"]
    target = full[y0:y1, x0:x1].copy()
    annotation = original_annotation(mask_original, image_label, mapping)
    result["metadata"]["input_source"] = {"adapter": "industrial-geometry-v1", "mapping": mapping,
        "original_mode": image.mode, "image_label_policy": "caller original label retained, including ROI-invisible defects",
        "color_processing": {"ICC_if_present": "LittleCMS to sRGB; relative-colorimetric intent=1; flags=0; RGB uint8",
                             "no_ICC": "assume standard sRGB explicitly", "inverse_transfer_count": 1,
                             "resize_and_synthesis_domain": "linear RGB", "output_encoding": "caller invokes linear_to_srgb once for detection/display"},
        "annotation": annotation["statistics"], "full_canvas_error_summary": full_result["metadata"]["error_summary"]}
    return result, target, annotation
