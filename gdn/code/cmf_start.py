"""Start-reference eight-increment CMF for a separately declared GDN adaptation.

No blur synthesis, image/label warp, augmentation, file writes or Torch import.
The half lattice carries full-resolution pixel displacements, without /2 or /16.
Crop origins are kept explicitly, including odd-offset training crops.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA = '18b50dbf8bd2d317c75c9ed557628d831ee22b483d33d8e33181fbf01228fd37'
PROTOCOL_SHA = '86778bc7cf0b77074e1e30846124cfd6ffd64417085b81a8bbdfb6f7944f3e86'

def _load_reference():
    path = ROOT / 'vendor/reference/imu_synthesis.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA:
        raise ValueError('Frozen rotation implementation differs')
    spec = importlib.util.spec_from_file_location('gdn_start_frozen_rotation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

rotation = _load_reference()

def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value

def padded_shape(height, width):
    _positive_int(height, 'height'); _positive_int(width, 'width')
    return 8*((height+7)//8), 8*((width+7)//8)

def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()

def start_cmf(k_roi, interval_times, rates_rad_s, height, width, *,
              origin_xy=(0, 0)):
    """Return padded H/2 x W/2 x 16 float32 CMF and an explicit contract.

    K is defined on the already frozen common ROI. origin_xy locates a crop
    within that ROI. Padding extends the prior analytically; RGB is zero-padded
    by the model adapter, and evaluation always removes padding.
    """
    k = rotation.validate_k(k_roi)
    times = rotation.validate_times_seconds(interval_times)
    rates = np.asarray(rates_rad_s, dtype=np.float64)
    if times.shape != (5,) or rates.shape != (4, 3) or not np.isfinite(rates).all():
        raise ValueError('Exactly four finite camera-body rate intervals required')
    if (not isinstance(origin_xy, (tuple, list)) or len(origin_xy) != 2
            or any(type(v) is not int or v < 0 for v in origin_xy)):
        raise ValueError('Crop origin must be two nonnegative integer ROI coordinates')
    hpad, wpad = padded_shape(height, width)
    nodes = np.linspace(times[0], times[-1], 9)
    poses = rotation.orientations_at(times, rates, nodes)
    u = origin_xy[0] + np.arange(0, wpad, 2, dtype=np.float64)
    v = origin_xy[1] + np.arange(0, hpad, 2, dtype=np.float64)
    points = np.stack(np.meshgrid(u, v), axis=-1)
    projected = []
    for pose in poses:
        h = rotation.forward_homography(k, pose, poses[0])
        xy, valid = rotation.project_points(h, points)
        if not valid.all() or not np.isfinite(xy).all():
            raise ValueError('Invalid projected prior on required padded lattice; no fallback')
        projected.append(xy)
    increments = np.diff(np.stack(projected), axis=0)
    cmf64 = increments.transpose(1, 2, 0, 3).reshape(hpad//2, wpad//2, 16)
    cmf = cmf64.astype(np.float32)
    if not np.isfinite(cmf).all():
        raise ValueError('Nonfinite CMF after float32 conversion')
    contract = {
        'schema': 'gdn-start-cmf-v1', 'target_reference': 'exposure_start',
        'adaptation': 'GDN architecture/common-budget; not native-center checkpoint input',
        'temporal_nodes_seconds': nodes.tolist(), 'interval_times_seconds': times.tolist(),
        'channel_order': [f'delta_{i}_{axis}' for i in range(8) for axis in ('x', 'y')],
        'units': 'full-resolution ROI pixel displacement; no divide by2 or16',
        'input_hw': [height, width], 'padded_rgb_hw': [hpad, wpad],
        'cmf_hwc': list(cmf.shape), 'lattice_origin_roi_xy': list(origin_xy),
        'lattice_stride_xy': [2, 2], 'rgb_padding': 'right/bottom zero to multiple8',
        'prior_padding': 'analytic projection at actual extended coordinates',
        'reference_code_sha256': REFERENCE_SHA,
        'cmf_float32_sha256': hashlib.sha256(cmf.tobytes(order='C')).hexdigest(),
        'numpy': np.__version__,
    }
    return cmf, contract

def from_q16_metadata(metadata, height, width, *, crop=None):
    """Read frozen C3 geometry only; caller retains the original B/GT/mask.

    This is not a replacement for Q2's full allocation/NPZ preflight. It checks
    local metadata identities and rejects implicit legacy or center conversion.
    """
    if metadata.get('schema_version') != 'imda-float-xy0-v2':
        raise ValueError('Explicit q16 float sample metadata required')
    if metadata.get('camera_parameter_origin') != 'simulated':
        raise ValueError('This fixed protocol requires simulated camera parameters')
    source = metadata['input_source']
    if source['protocol_sha256'] != PROTOCOL_SHA:
        raise ValueError('Sample is not from frozen q16 protocol')
    mapping = source['mapping']; traj = source['trajectory']
    if mapping['crop_size_wh'] != [width, height]:
        raise ValueError('Original RGB shape differs from common ROI metadata')
    rates = traj['rates_rad_s']
    if _hash_json(rates) != traj['rates_sha256']:
        raise ValueError('Recorded raw angular rates differ from their C3 identity')
    if crop is None:
        left, top, out_h, out_w = 0, 0, height, width
    else:
        if not isinstance(crop, (tuple, list)) or len(crop) != 4:
            raise ValueError('Crop must be top,left,height,width')
        top, left, out_h, out_w = crop
        if any(type(v) is not int for v in crop) or min(top, left) < 0:
            raise ValueError('Integer in-ROI crop required')
        _positive_int(out_h, 'crop height'); _positive_int(out_w, 'crop width')
        if top+out_h > height or left+out_w > width:
            raise ValueError('Crop exceeds original ROI; no resize or retry')
    return start_cmf(mapping['k_crop'], np.linspace(0., .02, 5), rates,
                     out_h, out_w, origin_xy=(left, top))
