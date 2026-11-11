#!/usr/bin/env python3
"""NEW 2026-11-05 reference synthesis; NOT the historical GAMD generator.

No import side effects or implicit writes. No real training/industrial validation;
legacy PSNR/SSIM cannot be attributed to this implementation.

Contract: pinhole/global shutter, static scene, fixed optical center, linear RGB
float image in [0,1]. Camera x right/y down/z forward. Q_WC maps camera to world.
K must be calibrated for the supplied image resolution/crop; it is not rescaled.
Already bias-corrected, camera-frame body gyro is rad/s; timestamps are seconds.
One rate per interval, held constant from its LEFT boundary. No implicit unit,
IMU extrinsic, bias, clock-offset, distortion, translation, or depth estimation.
Q_next = Q @ Exp([omega_C]x * dt); H_t0 = K @ Q_t.T @ Q_0 @ inv(K).

NEW network contract, NOT legacy BGR/PIL compatibility: nine float32 channels
[R_linear,G_linear,B_linear,C_dx,C_dy,0,E_dx,E_dy,0]. C/E are reference-pixel
displacements divided by explicit map_scale_px (default 1 pixel). No 127 offset,
JPEG, uint8 quantization, or displacement clipping. A corresponding loader and
new training/evaluation are required; historical checkpoints are not validated.
Masks are returned separately and are not silently appended to the nine channels.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def validate_k(k):
    k = np.asarray(k, dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("K must be a finite 3x3 calibrated pinhole matrix")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("K requires positive focal lengths and last row [0,0,1]")
    if abs(k[1, 0]) > 1e-12:
        raise ValueError("K must be upper triangular; skew K[0,1] is supported")
    return k


def validate_so3(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (3, 3) or not np.isfinite(q).all():
        raise ValueError("Q_WC must be finite 3x3")
    if not np.allclose(q.T @ q, np.eye(3), atol=1e-9, rtol=0) or not np.isclose(np.linalg.det(q), 1, atol=1e-9, rtol=0):
        raise ValueError("Q_WC must belong to SO(3), including determinant +1")
    return q


def validate_times_seconds(times_seconds):
    times = np.asarray(times_seconds, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("At least two finite, strictly increasing timestamps in seconds are required")
    return times


def exp_so3(rotation_vector_rad):
    v = np.asarray(rotation_vector_rad, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("rotation vector must have three finite radian components")
    x, y, z = v
    a = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    theta = float(np.linalg.norm(v))
    if theta < 1e-7:
        c1 = 1 - theta**2 / 6 + theta**4 / 120
        c2 = .5 - theta**2 / 24 + theta**4 / 720
    else:
        c1 = math.sin(theta) / theta
        c2 = (1 - math.cos(theta)) / theta**2
    return np.eye(3) + c1 * a + c2 * (a @ a)


def integrate_body_gyro(times_seconds, angular_rates_rad_s, q_wc_initial=None):
    times = validate_times_seconds(times_seconds)
    omega = np.asarray(angular_rates_rad_s, dtype=np.float64)
    if omega.shape != (len(times) - 1, 3) or not np.isfinite(omega).all():
        raise ValueError("Provide one finite camera-frame rad/s vector per time interval")
    q = validate_so3(np.eye(3) if q_wc_initial is None else q_wc_initial).copy()
    poses = [q.copy()]
    for duration, rate in zip(np.diff(times), omega):
        q = q @ exp_so3(rate * duration)
        poses.append(validate_so3(q).copy())
    return np.stack(poses)


def orientations_at(times_seconds, angular_rates_rad_s, sample_times_seconds, q_wc_initial=None):
    times = validate_times_seconds(times_seconds)
    poses = integrate_body_gyro(times, angular_rates_rad_s, q_wc_initial)
    samples = np.asarray(sample_times_seconds, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all() or np.any(samples < times[0]) or np.any(samples > times[-1]):
        raise ValueError("Sample times must lie inside the supplied exposure")
    indices = np.clip(np.searchsorted(times, samples, side="right") - 1, 0, len(times) - 2)
    return np.stack([poses[j] @ exp_so3(np.asarray(angular_rates_rad_s[j]) * (t - times[j]))
                     for j, t in zip(indices, samples)])


def forward_homography(k, q_wc_t, q_wc_0):
    k = validate_k(k)
    return k @ validate_so3(q_wc_t).T @ validate_so3(q_wc_0) @ np.linalg.inv(k)


def pixel_grid(height, width):
    if not isinstance(height, (int, np.integer)) or not isinstance(width, (int, np.integer)) or min(height, width) <= 0:
        raise ValueError("Image height/width must be positive integers")
    v, u = np.indices((height, width), dtype=np.float64)
    return np.stack([u, v], axis=-1)


def project_points(h, points):
    """Forward arbitrary pixel points; validity is positive depth, not image bounds."""
    h = np.asarray(h, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError("Finite 3x3 mapping and (...,2) points required")
    rays = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1) @ h.T
    valid = np.isfinite(rays).all(axis=-1) & (rays[..., 2] > 1e-12)
    mapped = np.full(points.shape, np.nan)
    np.divide(rays[..., :2], rays[..., 2, None], out=mapped, where=valid[..., None])
    return mapped, valid


def dense_forward_trajectories(k, orientations_wc, height, width):
    """Reference pixel centers -> time-t positions, float64, shape T,H,W,2."""
    poses = np.asarray(orientations_wc, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (3, 3) or len(poses) == 0:
        raise ValueError("Nonempty T,3,3 orientations required")
    grid = pixel_grid(height, width)
    values = [project_points(forward_homography(k, q, poses[0]), grid) for q in poses]
    return np.stack([x[0] for x in values]), np.stack([x[1] for x in values])


def exposure_quadrature(times_seconds, samples_per_interval=1):
    """Uniform-in-time midpoints within each interval; dt/(tau*n) weights."""
    times = validate_times_seconds(times_seconds)
    counts_raw = np.asarray(samples_per_interval)
    if counts_raw.ndim == 0:
        counts_raw = np.repeat(counts_raw, len(times) - 1)
    if counts_raw.shape != (len(times) - 1,) or not np.isfinite(counts_raw).all() or np.any(counts_raw < 1) or np.any(counts_raw != np.floor(counts_raw)):
        raise ValueError("Positive integer sample count required for every interval")
    sample_times, weights = [], []
    for start, duration, count in zip(times[:-1], np.diff(times), counts_raw.astype(int)):
        sample_times.extend(start + (np.arange(count) + .5) / count * duration)
        weights.extend(np.full(count, duration / ((times[-1] - times[0]) * count)))
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("Exposure quadrature is not normalized")
    return np.asarray(sample_times), weights


def inverse_sample_linear_rgb(image_linear_rgb, h_forward):
    """Inverse image warp with bilinear interpolation and zero outside the frame.

    A sample is valid only when its source center lies within [0,W-1]x[0,H-1]
    and in front of the reference camera. Boundary centers remain valid; adjacent
    edge indices are clamped only for bilinear evaluation at those exact edges.
    """
    image = np.asarray(image_linear_rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Image must be finite H,W,3 linear RGB in [0,1]")
    height, width = image.shape[:2]
    source, valid = project_points(np.linalg.inv(h_forward), pixel_grid(height, width))
    tolerance = 1e-9  # stabilize numerically identical mappings at image edges
    valid &= (source[..., 0] >= -tolerance) & (source[..., 0] <= width - 1 + tolerance)
    valid &= (source[..., 1] >= -tolerance) & (source[..., 1] <= height - 1 + tolerance)
    safe = np.where(valid[..., None], source, 0)
    x = np.clip(safe[..., 0], 0, width - 1)
    y = np.clip(safe[..., 1], 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = (x - x0)[..., None], (y - y0)[..., None]
    sampled = ((1-dx)*(1-dy)*image[y0, x0] + dx*(1-dy)*image[y0, x1]
               + (1-dx)*dy*image[y1, x0] + dx*dy*image[y1, x1])
    sampled[~valid] = 0
    return sampled.astype(np.float32), valid


def trapezoid_weights(times_seconds):
    times = validate_times_seconds(times_seconds)
    dt = np.diff(times)
    weights = np.r_[dt[0] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1] / 2]
    return weights / (times[-1] - times[0])


def encode_displacements(displacements_px, map_scale_px):
    """Float64 scaling -> float32; reject overflow, underflow, or >1e-6 loss.

    Scale itself must lie in float32's positive normal range. Every nonzero
    component must survive encoding and round-trip within relative 1e-6.
    This is a numerical interface bound, not a measured trajectory accuracy.
    """
    scale = float(map_scale_px)
    limits = np.finfo(np.float32)
    if not np.isfinite(scale) or not float(limits.tiny) <= scale <= float(limits.max):
        raise ValueError("map_scale_px must lie in float32 positive normal range")
    original = np.asarray(displacements_px, dtype=np.float64)
    if not np.isfinite(original).all():
        raise ValueError("Displacements must be finite before encoding")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        encoded = (original / scale).astype(np.float32)
        decoded = encoded.astype(np.float64) * scale
    nonzero = original != 0
    if not np.isfinite(encoded).all() or not np.isfinite(decoded).all():
        raise ValueError("Scale overflows encoded or decoded descriptors")
    if np.any(encoded[nonzero] == 0) or np.any(np.abs(decoded[nonzero]-original[nonzero]) > 1e-6*np.abs(original[nonzero])):
        raise ValueError("Scale loses nonzero displacement information beyond relative 1e-6")
    return encoded


def quadratic_errors(displacements_px, times_seconds, control_px, endpoint_px, fit_valid):
    """Per-pixel errors for supplied decoded control/end coordinates at fit times."""
    times = validate_times_seconds(times_seconds)
    s = (times-times[0])/(times[-1]-times[0])
    valid = np.asarray(fit_valid, dtype=bool)
    clean = np.where(valid[None, ..., None], displacements_px, 0)
    prediction = (2*s*(1-s))[:, None, None, None]*np.asarray(control_px, dtype=np.float64)
    prediction += (s*s)[:, None, None, None]*np.asarray(endpoint_px, dtype=np.float64)
    distances = np.sum((prediction-clean)**2, axis=-1)
    rms = np.sqrt(np.sum(trapezoid_weights(times)[:, None, None]*distances, axis=0))
    maximum = np.sqrt(distances.max(axis=0))
    rms[~valid], maximum[~valid] = np.nan, np.nan
    return rms.astype(np.float32), maximum.astype(np.float32)


def fit_dense_quadratic(displacements_px, times_seconds, trajectory_valid=None):
    """Fit beta(s)=2s(1-s)C+s^2 E with beta(0)=0, beta(1)=E.

    Actual s=(t-t0)/tau and trapezoidal time weights define least squares.
    Errors are measured at all supplied timestamps, not between timestamps.
    Trajectories invalid at any supplied fit node are excluded, mask=false.
    """
    times = validate_times_seconds(times_seconds)
    d = np.asarray(displacements_px, dtype=np.float64)
    if d.ndim != 4 or d.shape[0] != len(times) or d.shape[-1] != 2:
        raise ValueError("Displacements must have shape T,H,W,2")
    valid = np.isfinite(d).all(axis=-1)
    if trajectory_valid is not None:
        mask = np.asarray(trajectory_valid, dtype=bool)
        if mask.shape != d.shape[:-1]:
            raise ValueError("Trajectory mask must have shape T,H,W")
        valid &= mask
    valid = valid.all(axis=0)
    if np.any(np.abs(d[0][valid]) > 1e-8):
        raise ValueError("Displacements must start at zero")
    s = (times - times[0]) / (times[-1] - times[0])
    a, b, weights = 2*s*(1-s), s*s, trapezoid_weights(times)
    denominator = float(np.dot(weights, a*a))
    if denominator <= 0:
        raise ValueError("At least one interior timestamp is needed to determine control")
    clean = np.where(valid[None, ..., None], d, 0)
    end = clean[-1]
    residual = clean - b[:, None, None, None] * end
    control = np.sum((weights*a)[:, None, None, None] * residual, axis=0) / denominator
    # Report errors for the float32 descriptors actually returned, not only
    # their higher-precision least-squares intermediates.
    control = control.astype(np.float32)
    end = end.astype(np.float32)
    rms, maximum = quadratic_errors(d, times, control, end, valid)
    return {"control_px": control, "endpoint_px": end,
            "fit_valid": valid, "rms_error_px": rms.astype(np.float32),
            "max_error_px": maximum.astype(np.float32)}


def error_summary(fit):
    valid = fit["fit_valid"]
    result = {"valid_pixels": int(valid.sum()), "total_pixels": int(valid.size),
              "valid_fraction": float(valid.mean()), "scope": "all valid full-frame pixels; errors at supplied fit times only"}
    for name in ("rms_error_px", "max_error_px"):
        values = np.asarray(fit[name])[valid]
        result[name] = ({"mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
                         "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
                         "p99": float(np.percentile(values, 99)), "max": float(values.max())} if values.size else None)
    return result


def synthesize(image_linear_rgb, k, times_seconds, angular_rates_rad_s,
               samples_per_interval=2, q_wc_initial=None, map_scale_px=1.0, *,
               camera_parameter_origin=None, color_source=None):
    """Return new float artifacts/masks. Does not read or write any files."""
    k, times = validate_k(k), validate_times_seconds(times_seconds)
    image = np.asarray(image_linear_rgb, dtype=np.float64)
    # Validate before any substantive allocations; same image contract as sampler.
    inverse_sample_linear_rgb(image, np.eye(3))
    # Validate scale even for a zero-motion image.
    encode_displacements(np.zeros(1), map_scale_px)
    sample_times, weights = exposure_quadrature(times, samples_per_interval)
    poses = orientations_at(times, angular_rates_rad_s, sample_times, q_wc_initial)
    q0 = integrate_body_gyro(times, angular_rates_rad_s, q_wc_initial)[0]
    blurred = np.zeros_like(image)
    coverage = np.zeros(image.shape[:2], dtype=np.float64)
    for q, weight in zip(poses, weights):
        sampled, mask = inverse_sample_linear_rgb(image, forward_homography(k, q, q0))
        blurred += weight*sampled
        coverage += weight*mask
    # Include true endpoints and all midpoint samples; never fit only two ends.
    fit_times = np.unique(np.r_[times, sample_times])
    fit_poses = orientations_at(times, angular_rates_rad_s, fit_times, q_wc_initial)
    trajectories, trajectory_valid = dense_forward_trajectories(k, fit_poses, *image.shape[:2])
    displacements = trajectories - pixel_grid(*image.shape[:2])[None]
    fit = fit_dense_quadratic(displacements, fit_times, trajectory_valid)
    zeros = np.zeros(image.shape[:2] + (1,), dtype=np.float32)
    cp = np.concatenate([encode_displacements(fit["control_px"], map_scale_px), zeros], axis=-1)
    ep = np.concatenate([encode_displacements(fit["endpoint_px"], map_scale_px), zeros], axis=-1)
    fit["rms_error_px"], fit["max_error_px"] = quadratic_errors(
        displacements, fit_times, cp[..., :2].astype(np.float64)*float(map_scale_px),
        ep[..., :2].astype(np.float64)*float(map_scale_px), fit["fit_valid"])
    metadata = {"implementation": "NEW reference 2026-11-05; not historical GAMD",
                "empirical_status": "No real training or industrial validation; legacy PSNR/SSIM not applicable",
                "geometry": "fixed optical center; static scene; caller-supplied pinhole parameters; global shutter",
                "units": {"time": "seconds", "gyro": "camera-frame body rad/s, interval-left hold", "displacement": "pixels"},
                "network_channel_order": ["R_linear", "G_linear", "B_linear", "C_dx", "C_dy", "0", "E_dx", "E_dy", "0"],
                "map_scale_px": float(map_scale_px), "legacy_compatible": False,
                "descriptor_encoding": "float64 divide then float32; normal-range scale; relative round-trip <=1e-6; no nonzero underflow",
                "error_basis": "actual float32 cp/ep maps decoded with map_scale_px at fit_times_seconds",
                "loader_requirement": "float32 new xy0 map order; masks separate; corresponding training/evaluation required",
                "boundary": "bilinear inverse sampling; outside zero; coverage is weighted validity at supplied quadrature nodes; no per-pixel renormalization",
                "mask_scope": "fit_valid: supplied fit nodes; full_exposure_valid: weighted coverage within absolute 1e-7 of one at supplied quadrature nodes, not a continuous-exposure guarantee",
                "error_summary": error_summary(fit), "exposure_weight_sum": float(weights.sum())}
    if camera_parameter_origin is not None:
        if camera_parameter_origin not in ("simulated", "calibrated"):
            raise ValueError("Explicit camera_parameter_origin must be simulated or calibrated")
        metadata["camera_parameter_origin"] = camera_parameter_origin
    if color_source is not None:
        metadata["color_source"] = dict(color_source)
    return {"blur_linear_rgb": blurred.astype(np.float32), "cp_map": cp, "ep_map": ep,
            "network_input_9": np.concatenate([blurred.astype(np.float32), cp, ep], axis=-1),
            "exposure_coverage": coverage.astype(np.float32), "full_exposure_valid": np.isclose(coverage, 1, atol=1e-7, rtol=0),
            "sample_times_seconds": sample_times, "exposure_weights": weights,
            "fit_times_seconds": fit_times, "trajectory_valid": trajectory_valid,
            **fit, "metadata": metadata}


def srgb_to_linear(image_rgb):
    image = np.asarray(image_rgb, dtype=np.float64)
    return np.where(image <= .04045, image / 12.92, ((image + .055) / 1.055)**2.4).astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="RGB image decoded by PIL, assumed standard sRGB")
    parser.add_argument("--imu-json", type=Path, help="K, times_seconds, angular_rates_rad_s; optional q_wc_initial")
    parser.add_argument("--samples-per-interval", type=int, default=2)
    parser.add_argument("--map-scale-px", type=float, default=1.)
    parser.add_argument("--write", type=Path, help="Only explicit output switch; new lossless .npz, refuses overwrite")
    args = parser.parse_args(argv)
    if bool(args.image) != bool(args.imu_json):
        parser.error("Provide --image and --imu-json together, or neither for a synthetic demonstration")
    if args.image:
        with Image.open(args.image) as source:
            image = srgb_to_linear(np.asarray(source.convert("RGB"), dtype=np.float64)/255.)
        configuration = json.loads(args.imu_json.read_text())
        k, times, omega = (configuration[key] for key in ("K", "times_seconds", "angular_rates_rad_s"))
        q_initial = configuration.get("q_wc_initial")
        source_kind = "user supplied image/configuration; not an evaluated benchmark"
    else:
        grid = pixel_grid(9, 13)
        image = np.stack([grid[..., 0]/12, grid[..., 1]/8, np.full((9, 13), .4)], axis=-1)
        k = [[20, 0, 6], [0, 15, 4], [0, 0, 1]]
        times, omega, q_initial = [0, .01, .04], [[.1, .3, .2], [-.2, .1, .4]], None
        source_kind = "invented 9x13 demonstration; no empirical performance claim"
    result = synthesize(image, k, times, omega, args.samples_per_interval, q_initial, args.map_scale_px)
    result["metadata"]["input_source"] = source_kind
    if args.write is not None:
        if args.write.suffix != ".npz":
            parser.error("--write requires a .npz filename; no JPEG output is supported")
        # Opening with xb prevents accidental overwrite. No mkdir/delete on import
        # or on the default validation/reporting path. Parent must already exist.
        with args.write.open("xb") as output:
            np.savez_compressed(output, **{key: value for key, value in result.items() if key != "metadata"},
                                metadata_json=np.array(json.dumps(result["metadata"], ensure_ascii=False)))
    print(json.dumps(result["metadata"], indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
