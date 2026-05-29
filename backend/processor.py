# Improves MediaPipe landmark data before BVH conversion.

import numpy as np
from scipy.signal import savgol_filter

# MediaPipe landmark indices we care about
IDX_LEFT_HIP      = 23
IDX_RIGHT_HIP     = 24
IDX_LEFT_SHOULDER = 11
IDX_RIGHT_SHOULDER= 12

def frames_to_array(frames: list, use_world: bool = True) -> np.ndarray:
    """
    Convert list of frame dicts to numpy array.
    Shape: (num_frames, 33, 4)  — 33 landmarks, x/y/z/visibility
    """
    n = len(frames)
    arr = np.zeros((n, 33, 4), dtype=np.float32)
    for i, frame in enumerate(frames):
        src = frame.worldLandmarks if (use_world and frame.worldLandmarks) else frame.landmarks
        for j, lm in enumerate(src):
            arr[i, j, 0] = lm.x
            arr[i, j, 1] = lm.y
            arr[i, j, 2] = lm.z
            arr[i, j, 3] = lm.v
    return arr


def array_to_world_landmarks(arr: np.ndarray) -> list[list[dict]]:
    """
    Convert numpy array back to list of worldLandmarks lists.
    """
    result = []
    for i in range(arr.shape[0]):
        frame_lms = []
        for j in range(33):
            frame_lms.append({
                "x": float(arr[i, j, 0]),
                "y": float(arr[i, j, 1]),
                "z": float(arr[i, j, 2]),
                "v": float(arr[i, j, 3]),
            })
        result.append(frame_lms)
    return result


def smooth_landmarks(arr: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter along the time axis for each landmark and axis.
    Much better than a simple moving average — preserves peaks (fast movements)
    while removing jitter.
    Window must be odd and > poly. We shrink it if there aren't enough frames.
    """
    n_frames = arr.shape[0]
    # Need at least window frames; shrink window if clip is short
    w = min(window, n_frames if n_frames % 2 == 1 else n_frames - 1)
    w = max(w, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    if w > n_frames:
        # Not enough frames to smooth, return as-is
        return arr

    smoothed = arr.copy()
    for joint in range(33):
        for axis in range(3):  # Only smooth x, y, z
            smoothed[:, joint, axis] = savgol_filter(
                arr[:, joint, axis], window_length=w, polyorder=poly
            )
    return smoothed


def correct_hip_orientation(arr: np.ndarray) -> np.ndarray:
    """
    Heuristic hip/spine orientation correction for when a person turns
    away from the camera.

    MediaPipe's worldLandmarks z values become unreliable when the person
    is not facing the camera. We detect this by comparing the apparent
    shoulder width to a reference, then correct the hip z-rotation.

    This is a heuristic — it won't be perfect for orbiting cameras, but
    it significantly improves single-axis turns (person rotating in place).
    """
    corrected = arr.copy()
    n = arr.shape[0]

    # ── Measure reference shoulder width from the most frontal frames ────────
    shoulder_widths = []
    for i in range(n):
        ls = arr[i, IDX_LEFT_SHOULDER, :3]
        rs = arr[i, IDX_RIGHT_SHOULDER, :3]
        # Only x/y width — what the camera "sees"
        w = abs(rs[0] - ls[0])
        shoulder_widths.append(w)

    shoulder_widths = np.array(shoulder_widths)
    # Use 90th percentile as reference (avoids outliers from arms being raised)
    ref_width = np.percentile(shoulder_widths, 90)
    if ref_width < 1e-4:
        return corrected

    for i in range(n):
        ls = corrected[i, IDX_LEFT_SHOULDER, :3]
        rs = corrected[i, IDX_RIGHT_SHOULDER, :3]
        lh = corrected[i, IDX_LEFT_HIP, :3]
        rh = corrected[i, IDX_RIGHT_HIP, :3]

        apparent_width = abs(rs[0] - ls[0])
        # Ratio of 1.0 = fully facing camera, ~0 = fully turned away
        facing_ratio = min(apparent_width / ref_width, 1.0)

        if facing_ratio > 0.85:
            # No correction needed, person is mostly facing camera
            continue

        # Estimate rotation angle from the facing ratio
        turn_angle = np.arccos(np.clip(facing_ratio, 0.0, 1.0))

        # Determine turn direction
        ls_z = ls[2]
        rs_z = rs[2]
        turning_left = rs_z < ls_z

        # Apply z-correction to hip and shoulder landmarks
        hip_mid_x = (lh[0] + rh[0]) / 2
        sho_mid_x = (ls[0] + rs[0]) / 2

        # Nudge the spine z to reflect the turn
        z_offset = np.sin(turn_angle) * ref_width * 0.5
        if not turning_left:
            z_offset = -z_offset

        # Apply to hips and shoulders proportionally — don't move ankles/wrists
        spine_joints = [
            IDX_LEFT_HIP, IDX_RIGHT_HIP,
            IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER,
            11, 12, 23, 24,  # same as above, included for clarity
        ]
        # Also nudge mid-body landmarks
        mid_body = [13, 14, 23, 24, 11, 12]  # Elbows, hips, shoulders
        for idx in set(mid_body):
            corrected[i, idx, 2] += z_offset * 0.6

    return corrected


def process_frames(frames: list) -> list:
    """
    Main entry point. Takes a list of Frame pydantic objects,
    returns a list of dicts ready to be sent back as JSON.
    """
    # Convert to numpy for fast processing
    arr = frames_to_array(frames, use_world=True)

    # Smooth jitter
    arr = smooth_landmarks(arr)

    # Hip orientation heuristic
    arr = correct_hip_orientation(arr)

    # Convert back and rebuild frame dicts
    improved_world = array_to_world_landmarks(arr)

    result = []
    for i, frame in enumerate(frames):
        result.append({
            "frameIndex": frame.frameIndex,
            "timeMs": frame.timeMs,
            "landmarks": [
                {"x": lm.x, "y": lm.y, "z": lm.z, "v": lm.v}
                for lm in frame.landmarks
            ],
            "worldLandmarks": improved_world[i],
        })
    return result