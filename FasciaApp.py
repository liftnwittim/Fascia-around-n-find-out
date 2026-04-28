# CoreDataContract

import cv2
import numpy as np
from scipy.stats import pearsonr
import mediapipe as mp
import os
from flask import Flask, request, jsonify

from flask import Flask, request, jsonify
app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class FramePacket:
    rgb_frames: List[np.ndarray] = field(default_factory=list)
    skeleton: Dict[str, Any] = field(default_factory=dict)
    skin_mask: np.ndarray = None
    fps: int = 30
    timestamp_ms: int = 0

@dataclass
class ModuleScore:
    raw: float = 0.0
    confidence: float = 0.0
    flags: List[Any] = field(default_factory=list)
    debug_map: np.ndarray = None
@dataclass
class AlertFlag:
    code: str = ""
    region: str = ""
    severity: str = "MEDIUM"
    message: str = ""

@dataclass
class FascialIntegrityScore:
    score: float = 0.0
    FEI: float = 0.0
    tier: str = ""
    alert_queue: List[Any] = field(default_factory=list)
    module_breakdown: Dict[str, float] = field(default_factory=dict)

# ShearingForceAlgorithm - measures optical flow

def shearing_force(packet: FramePacket) -> ModuleScore:

    # ── 1. Extract region of interest ──────────────────────────────
    # Thoracolumbar: bounding box from left_hip ↔ right_hip ↔ left_shoulder
    ROI = crop_between_landmarks(
        packet.skeleton,
        anchors=["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
        padding_px=40
    )

    # ── 2. Compute dense optical flow (skin surface) ────────────────
    # Farneback on skin-masked region only, suppress bone-adjacent pixels
    surface_flow: np.ndarray = []  # shape (T-1, ROI_H, ROI_W, 2)

    for t in range(len(ROI_frames) - 1):
        frame_a = apply_mask(ROI_frames[t],   skin_mask_roi)
        frame_b = apply_mask(ROI_frames[t+1], skin_mask_roi)
        flow = cv2.calcOpticalFlowFarneback(
            gray(frame_a), gray(frame_b),
            None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        surface_flow.append(flow)

    # ── 3. Extract rigid-body motion from skeleton (bone-proxy) ────
    # This is the "true" segment motion you subtract out
    bone_vectors: List[Vec2] = []
    for t in range(len(packet.skeleton_frames) - 1):
        delta = skeleton_delta_2d(packet.skeleton_frames[t],
                                  packet.skeleton_frames[t+1],
                                  region="thoracolumbar")
        bone_vectors.append(delta)

    # ── 4. Residual = skin motion MINUS bone motion ─────────────────
    # High residual = fascia gliding freely (good)
    # Low residual  = skin locked to bone (densification flag)
    residual_map = []
    for t, flow in enumerate(surface_flow):
        bone_field = broadcast_vector(bone_vectors[t], flow.shape)
        residual = flow - bone_field          # vector subtraction
        residual_magnitude = np.linalg.norm(residual, axis=-1)  # scalar map
        residual_map.append(residual_magnitude)

    mean_shear = np.mean(residual_map)         # mm/frame, calibrate per px-to-mm ratio

    # ── 5. Regional collapse detection ─────────────────────────────
    # Divide thoracolumbar into 3 vertical bands: upper / mid / lower
    band_scores = compute_band_means(residual_map, bands=3)

    flags = []
    # 30% threshold from spec — flag if any band falls 30% below baseline
    BASELINE_SHEAR_MM = 2.0   # healthy reference from calibration dataset
    for band, score in enumerate(band_scores):
        if score < BASELINE_SHEAR_MM * 0.70:
            flags.append(AlertFlag(
                code="FASCIAL_DENSIFICATION",
                region=f"thoracolumbar_band_{band}",
                severity="HIGH" if score < BASELINE_SHEAR_MM * 0.50 else "MEDIUM",
                message="Stagnant fascia / potential trigger point zone"
            ))

    normalized_score = sigmoid_normalize(mean_shear, lo=0.3, hi=3.5)

    return ModuleScore(
        raw=normalized_score,
        confidence=flow_quality_estimate(surface_flow),
        flags=flags,
        debug_map=np.mean(residual_map, axis=0)
    )

# Foot-to-Glute Chain 'Hyperarch Connectivity'

def foot_glute_chain(packet: FramePacket) -> ModuleScore:

    flags = []

    # ── 1. Arch height index (AHI) ──────────────────────────────────
    # Use landmark geometry: navicular height / foot length
    heel    = packet.skeleton["right_heel"]
    toe     = packet.skeleton["right_foot_index"]
    navicular = interpolate_landmark(heel, toe, t=0.35)  # ~35% from heel

    foot_length_px  = euclidean(heel, toe)
    arch_height_px  = navicular.y - heel.y   # positive = arch present
    AHI             = arch_height_px / foot_length_px

    ARCH_THRESHOLD  = 0.16   # AHI below this = collapsed arch
    arch_collapsed  = AHI < ARCH_THRESHOLD

    # ── 2. ATT visibility detection ─────────────────────────────────
    # Anterior tibial tendon runs medial to lateral malleolus up the shin
    # We look for a high-gradient ridge in the skin texture in that zone
    shin_roi  = crop_shin_region(packet.rgb_frames[-1], packet.skeleton)
    edges     = cv2.Sobel(gray(shin_roi), cv2.CV_64F, dx=1, dy=0, ksize=3)
    # ATT is a vertical ridge — sum vertical gradient column through tendon column
    col_range = get_att_column(packet.skeleton)   # x-range based on landmarks
    att_ridge_strength = np.mean(np.abs(edges[:, col_range]))

    ATT_THRESHOLD = 12.0     # calibrated gradient magnitude units
    att_visible   = att_ridge_strength > ATT_THRESHOLD

    # ── 3. Glute activation proxy ───────────────────────────────────
    # Cannot EMG, but we can detect glute surface displacement during
    # weight-bearing. High glute activation = posterior superior iliac spine
    # (PSIS) region shows tensioning micro-displacement on weight load.
    glute_shear = compute_regional_shear(packet, region="PSIS_bilateral")

    # ── 4. Connectivity index ───────────────────────────────────────
    # All three must be active simultaneously for true chain integrity
    connectivity_score = (
        normalize(AHI,              lo=0.10, hi=0.30) * 0.35 +
        normalize(att_ridge_strength, lo=6, hi=20)   * 0.35 +
        normalize(glute_shear,        lo=0, hi=2.5)  * 0.30
    )

    if arch_collapsed:
        flags.append(AlertFlag(
            code="ARCH_COLLAPSE",
            region="foot_right",
            severity="HIGH",
            message="Arch collapse detected — fascial disconnection, glutes likely disengaged"
        ))
    if not att_visible:
        flags.append(AlertFlag(
            code="ATT_NOT_ENGAGED",
            region="anterior_shin",
            severity="MEDIUM",
            message="ATT not prominent — proprioceptive drive from foot is compromised"
        ))
    if arch_collapsed and not att_visible:
        connectivity_score *= 0.4   # chain broken at source: hard penalty

    return ModuleScore(
        raw=connectivity_score,
        confidence=landmark_visibility_score(packet.skeleton, ["right_heel","right_foot_index"]),
        flags=flags,
        debug_map=edges
    )

# Movement Bandwith 

def movement_bandwidth(packet: FramePacket) -> ModuleScore:

    flags = []

    # ── 1. Extract rotation time-series for upper and lower body ───
    # Upper: vector from left_shoulder → right_shoulder → angle vs. frontal plane
    # Lower: vector from left_hip → right_hip → angle vs. frontal plane

    upper_angles: List[float] = []
    lower_angles: List[float] = []

    for skel in packet.skeleton_frames:
        upper_angles.append(compute_segment_angle(
            skel["left_shoulder"], skel["right_shoulder"], plane="transverse"))
        lower_angles.append(compute_segment_angle(
            skel["left_hip"], skel["right_hip"], plane="transverse"))

    # ── 2. Cross-correlation lag ────────────────────────────────────
    # Find the time offset at which upper_angles best matches lower_angles.
    # Zero lag = integrated tensegrity system (fascia transferring load instantly)
    # Positive lag = lower body moves first, upper catches up = segmented

    corr     = np.correlate(upper_angles - np.mean(upper_angles),
                            lower_angles - np.mean(lower_angles), mode='full')
    lag_idx  = np.argmax(corr) - (len(lower_angles) - 1)
    lag_ms   = (lag_idx / packet.fps) * 1000.0

    # ── 3. Amplitude ratio ──────────────────────────────────────────
    # Elite tensegrity: upper ROM ≈ lower ROM (load is distributed)
    # Poor integration: lower ROM >> upper ROM (hips dumping, no transfer)
    upper_ROM = np.max(upper_angles) - np.min(upper_angles)
    lower_ROM = np.max(lower_angles) - np.min(lower_angles)
    amp_ratio = min(upper_ROM, lower_ROM) / (max(upper_ROM, lower_ROM) + 1e-6)

    # ── 4. Spectral bandwidth ───────────────────────────────────────
    # Real fascia moves at multiple frequencies simultaneously (like a web).
    # Stiff or segmented movement is dominated by a single frequency.
    freqs_u = np.fft.rfft(upper_angles)
    freqs_l = np.fft.rfft(lower_angles)
    bandwidth_u = spectral_entropy(np.abs(freqs_u))
    bandwidth_l = spectral_entropy(np.abs(freqs_l))
    bandwidth_score = (bandwidth_u + bandwidth_l) / 2.0

    # ── 5. Composite ───────────────────────────────────────────────
    lag_score    = sigmoid_normalize(-abs(lag_ms), lo=-200, hi=0)   # penalize lag
    tensegrity_score = (
        lag_score       * 0.45 +
        amp_ratio       * 0.30 +
        bandwidth_score * 0.25
    )

    LAG_THRESHOLD_MS = 80.0
    if abs(lag_ms) > LAG_THRESHOLD_MS:
        flags.append(AlertFlag(
            code="SEGMENTED_MOVEMENT",
            severity="HIGH" if abs(lag_ms) > 150 else "MEDIUM",
            message=f"Upper/lower body lag = {lag_ms:.0f}ms — poor elastic recoil, injury risk on explosive movements"
        ))

    return ModuleScore(
        raw=tensegrity_score,
        confidence=visibility_confidence(packet, joints=["shoulders","hips"]),
        flags=flags,
        debug_map=generate_lag_heatmap(lag_ms, packet.skeleton)
    )

# Hydraulic/Thermal Indicator

def hydraulic_thermal(packet: FramePacket) -> ModuleScore:

    flags = []

    # ── 1. Skin color shift as temperature proxy ────────────────────
    # Compare red-channel mean of exposed skin (arms, neck, face) at
    # t=0 vs t=N (after warmup window). More red = more blood flow
    # = hyaluronic acid thinning = fascia becoming more mobile.

    skin_frames_start = packet.rgb_frames[:10]    # first ~2.5s at 4fps
    skin_frames_end   = packet.rgb_frames[-10:]   # last ~2.5s

    def mean_skin_red(frames):
        reds = []
        for f in frames:
            masked = f[:, :, 0][packet.skin_mask > 0]  # R channel, skin only
            reds.append(np.mean(masked))
        return np.mean(reds)

    red_start = mean_skin_red(skin_frames_start)
    red_end   = mean_skin_red(skin_frames_end)

    delta_red = red_end - red_start             # unsigned 8-bit units
    pct_change = delta_red / (red_start + 1e-6)

    # ── 2. Motion texture entropy shift ────────────────────────────
    # Cold, viscous fascia = stiff, low-entropy movement texture.
    # Warm fascia = richer micro-oscillation in skin surface.
    texture_start = mean_surface_entropy(packet.rgb_frames[:20], packet.skin_mask)
    texture_end   = mean_surface_entropy(packet.rgb_frames[-20:], packet.skin_mask)
    entropy_delta = texture_end - texture_start

    # ── 3. Score ───────────────────────────────────────────────────
    COLOR_THRESHOLD  = 0.03    # 3% red-channel increase = minimal adequate warmup
    ENTROPY_THRESHOLD = 0.05

    warmup_adequate = (pct_change > COLOR_THRESHOLD) and (entropy_delta > ENTROPY_THRESHOLD)

    hydro_score = (
        sigmoid_normalize(pct_change,     lo=0.0, hi=0.10) * 0.60 +
        sigmoid_normalize(entropy_delta,  lo=0.0, hi=0.15) * 0.40
    )

    if not warmup_adequate:
        flags.append(AlertFlag(
            code="VISCOSITY_LOCK",
            severity="HIGH",
            message="Hydraulic system still viscous — hyaluronic acid not thinned. Do NOT attempt explosive loading."
        ))

    return ModuleScore(
        raw=hydro_score,
        confidence=skin_coverage_ratio(packet.skin_mask),
        flags=flags,
        debug_map=generate_delta_heatmap(red_start, red_end, packet.skin_mask)
    )

# Stability Map

def stability_map(packet: FramePacket) -> ModuleScore:

    flags = []

    # ── 1. Center of mass trajectory ───────────────────────────────
    CoM_trace: List[Vec3] = []
    for skel in packet.skeleton_frames:
        # Weighted sum of major segment CoMs (head, torso, pelvis, limbs)
        CoM = compute_whole_body_CoM(skel, segment_weights=ANTHROPOMETRIC_TABLE)
        CoM_trace.append(CoM)

    CoM_trace = np.array(CoM_trace)   # shape (T, 3)

    # ── 2. Velocity and acceleration vectors ───────────────────────
    velocity     = np.gradient(CoM_trace, axis=0) * packet.fps
    acceleration = np.gradient(velocity,  axis=0) * packet.fps

    # ── 3. Force dispersion index ───────────────────────────────────
    # True tensegrity: acceleration is distributed across X/Y/Z equally.
    # Single-joint loading: one axis dominates (e.g., Z-axis spike = knee dumping).
    acc_var = np.var(acceleration, axis=0)   # [var_x, var_y, var_z]
    dispersion_index = np.min(acc_var) / (np.max(acc_var) + 1e-6)
    # 1.0 = perfectly distributed; ~0 = single-axis dominated

    # ── 4. Multi-joint coupling analysis ───────────────────────────
    # For each lower-limb joint pair (hip-knee, knee-ankle), compute
    # synchrony of angular velocity — high coupling = shared load (good)
    # low coupling = isolated joint absorbing = injury risk
    coupling_scores = []
    joint_pairs = [("left_hip","left_knee"), ("left_knee","left_ankle"),
                   ("right_hip","right_knee"), ("right_knee","right_ankle")]

    for j1, j2 in joint_pairs:
        ang_vel_1 = joint_angular_velocity(packet.skeleton_frames, j1)
        ang_vel_2 = joint_angular_velocity(packet.skeleton_frames, j2)
        r, _ = pearsonr(ang_vel_1, ang_vel_2)
        coupling_scores.append(max(r, 0))

    mean_coupling = np.mean(coupling_scores)

    # ── 5. CoM spike detection (injury flag) ───────────────────────
    acc_magnitudes = np.linalg.norm(acceleration, axis=-1)
    SPIKE_THRESHOLD = 3.5   # standard deviations above mean
    spike_mask = acc_magnitudes > (np.mean(acc_magnitudes) +
                                   SPIKE_THRESHOLD * np.std(acc_magnitudes))
    spike_count = np.sum(spike_mask)
    spike_joints = find_max_loading_joint(packet, spike_mask)

    variability_score = (
        dispersion_index  * 0.40 +
        mean_coupling     * 0.40 +
        normalize(1.0 / (spike_count + 1), lo=0, hi=1) * 0.20
    )

    if spike_count > 3:
        flags.append(AlertFlag(
            code="FORCE_CONCENTRATION",
            region=spike_joints,
            severity="HIGH",
            message=f"Force concentrated at {spike_joints} — {spike_count} CoM spikes. Fascial compartments not distributing load."
        ))
    if dispersion_index < 0.25:
        flags.append(AlertFlag(
            code="SINGLE_AXIS_LOADING",
            severity="MEDIUM",
            message="Uniplanar compensation pattern — variability tolerance low."
        ))

    return ModuleScore(
        raw=variability_score,
        confidence=skel_tracking_confidence(packet),
        flags=flags,
        debug_map=draw_CoM_trace(CoM_trace, spike_mask)
    )

# Pixel-to-Kinectic Compositor - Fascial Elasticity Index

# Module weights — reflect biological hierarchy of the chain
MODULE_WEIGHTS = {
    "shear":       0.25,   # M1: tissue quality ground truth
    "foot_glute":  0.20,   # M2: chain origin — garbage in, garbage out
    "tensegrity":  0.25,   # M3: the elastic recoil event itself
    "hydraulic":   0.15,   # M4: pre-condition gate
    "stability":   0.15    # M5: dispersion under load
}

def compute_FIS(modules: Dict[str, ModuleScore]) -> FascialIntegrityScore:

    # ── Gate check: hydraulic must clear first ──────────────────────
    # Cold fascia invalidates tensegrity and stability readings.
    # If M4 < 0.30, cap total score at 45 regardless of mechanics.
    hydraulic_gate = modules["hydraulic"].raw > 0.30

    # ── Confidence-weighted average ─────────────────────────────────
    weighted_sum  = 0.0
    weight_total  = 0.0

    for name, weight in MODULE_WEIGHTS.items():
        m = modules[name]
        effective_weight = weight * m.confidence   # low confidence = down-weighted
        weighted_sum  += m.raw * effective_weight
        weight_total  += effective_weight

    FEI_raw = weighted_sum / (weight_total + 1e-9)

    # ── Temporal smoothing (exponential moving average) ─────────────
    ALPHA = 0.25    # smoothing factor — resist single-frame noise
    FEI_smoothed = alpha_smooth(FEI_raw, previous_FEI, alpha=ALPHA)

    # ── Hydraulic gate enforcement ───────────────────────────────────
    FEI_final = FEI_smoothed if hydraulic_gate else min(FEI_smoothed, 0.45)

    # ── Scale to 0–100 and tier classify ───────────────────────────
    score_100 = round(FEI_final * 100, 1)

    tier = (
        "ELITE"         if score_100 >= 85 else
        "FUNCTIONAL"    if score_100 >= 60 else
        "COMPENSATING"  if score_100 >= 35 else
        "NO_BUENO"
    )

    # ── Aggregate alert queue ────────────────────────────────────────
    all_flags  = []
    for m in modules.values():
        all_flags.extend(m.flags)

    all_flags.sort(key=lambda f: {"HIGH":0,"MEDIUM":1,"LOW":2}[f.severity])

    return FascialIntegrityScore(
        score=score_100,
        FEI=FEI_final,
        tier=tier,
        alert_queue=all_flags,
        module_breakdown={k: round(v.raw * 100, 1) for k, v in modules.items()}
    )

# No bueno Alert System

ALERT_REGISTRY = {

    "FASCIAL_DENSIFICATION": {
        "trigger":    "M1 band shear < 70% of baseline",
        "risk":       "Trigger point formation, nerve entrapment",
        "action":     "Myofascial release protocol before loading",
        "retest_sec": 120
    },

    "ARCH_COLLAPSE": {
        "trigger":    "AHI < 0.16 under bodyweight",
        "risk":       "Glute inhibition, knee valgus cascade, plantar fascia load",
        "action":     "Short foot activation drill, check footwear heel drop",
        "retest_sec": 60
    },

    "ATT_NOT_ENGAGED": {
        "trigger":    "Anterior tibial ridge gradient < threshold",
        "risk":       "Proprioceptive deficit at foot — chain starts dead",
        "action":     "Heel walk × 20 reps to wake ATT, then retest",
        "retest_sec": 90
    },

    "SEGMENTED_MOVEMENT": {
        "trigger":    "Upper/lower rotational lag > 80ms",
        "risk":       "Shear force at lumbar during rotation, ACL stress",
        "action":     "Spiral-line integration drills (wood chops, rotational med ball)",
        "retest_sec": 180
    },

    "VISCOSITY_LOCK": {
        "trigger":    "Red-channel delta < 3% after warmup window",
        "risk":       "Hyaluronic acid too thick — micro-tears on ballistic load",
        "action":     "Add 5 min of dynamic warmup. NO explosive work until cleared.",
        "retest_sec": 300,
        "hard_block": True    # blocks explosive tier recommendations
    },

    "FORCE_CONCENTRATION": {
        "trigger":    "CoM spike > 3.5σ at single joint > 3 occurrences",
        "risk":       "Tendon/ligament overload — compensatory pattern entrenched",
        "action":     "Reduce load, address mobility deficit at identified joint",
        "retest_sec": 240
    },

    "SINGLE_AXIS_LOADING": {
        "trigger":    "Acceleration dispersion index < 0.25",
        "risk":       "Sagittal dominance — frontal/transverse plane blind spots",
        "action":     "Lateral + rotational loading — lateral band walks, step-overs",
        "retest_sec": 180
    }
}

def dispatch_alerts(alert_queue: List[AlertFlag], score: FascialIntegrityScore):
    for flag in alert_queue:
        meta = ALERT_REGISTRY[flag.code]
        yield {
            "severity":  flag.severity,
            "code":      flag.code,
            "region":    flag.region,
            "message":   flag.message,
            "risk":      meta["risk"],
            "action":    meta["action"],
            "retest_in": meta["retest_sec"],
            "hard_block": meta.get("hard_block", False)
        }

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if 'frame' not in request.files:
            return jsonify({"error": "No frame received"}), 400

        file = request.files['frame']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode frame"}), 400

        height, width = frame.shape[:2]

        return jsonify({
            "score": 0,
            "tier": "pending",
            "frame_received": True,
            "resolution": f"{width}x{height}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
