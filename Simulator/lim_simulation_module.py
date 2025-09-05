import math
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# =========================
# Utility / Math helpers
# =========================

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def unit(v):
    n = np.linalg.norm(v)
    if n == 0:
        return np.array([0.0, 0.0])
    return v / n

def softmax(x, temp=1.0):
    x = np.asarray(x, dtype=float)
    x = x / max(1e-8, temp)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def geometric_mean(vals, eps=1e-9):
    vals = np.asarray(vals, dtype=float)
    vals = np.clip(vals, eps, 1.0)
    return float(np.exp(np.mean(np.log(vals))))


# =========================
# Data schema constants
# =========================

# Minimal columns we expect after joining events with synthetic sidecar.
REQ_COLS = [
    "match_id","event_id","team_name","from_player_id","from_player_name",
    "x_location_start","y_location_start","x_location_end","y_location_end",
    "event_type","event","pressure","half_time",
    # Synthetic fatigue & effective kinematics
    "syn_f_i","v_i_eff","a_i_eff","d_i_eff","tau_i_react_eff","tau_i_turn_eff",
]

# Optional columns we try to use if present
OPT_COLS = [
    "line_break_direction","line_break_outcome","body_type","style","style_additional",
    "event_end_time_in_ms","match_run_time_in_ms"
]


# =========================
# State & config dataclasses
# =========================

@dataclass
class FieldConfig:
    length_m: float = 105.0
    width_m: float = 68.0
    nx: int = 36
    ny: int = 24

    def cell_size(self):
        return self.length_m / self.nx, self.width_m / self.ny


@dataclass
class BallModel:
    kappa_ground: float = 1.3  # s^-1 default
    ctrl_delay_base: float = 0.18
    ctrl_delay_pressure_mult: float = 1.35

    def control_delay(self, under_pressure: bool) -> float:
        return self.ctrl_delay_base * (self.ctrl_delay_pressure_mult if under_pressure else 1.0)


@dataclass
class PolicyModel:
    # π(a|zone,phase) as categorical over ["pass","carry","takeon","shot","clear"]
    actions: List[str] = field(default_factory=lambda: ["pass","carry","takeon","shot","clear"])
    zone_phase_probs: Dict[Tuple[int,int,str], np.ndarray] = field(default_factory=dict)  # (ix,iy,phase)->probs
    temperature: float = 1.0  # can be tuned to match entropy


@dataclass
class SuccessModel:
    # Simple linear logits per action type with guards and bins
    params: Dict[str, Dict[str, float]] = field(default_factory=dict)  # action -> coef dict
    # Example keys: {"pass":{"b0":..., "b_d":..., "b_press":..., "b_aerial":...}, ...}
    # Bounds
    p_min: float = 0.05
    p_max: float = 0.98


@dataclass
class MatchFit:
    field: FieldConfig
    ball: BallModel
    policy: PolicyModel
    success: SuccessModel
    grid_centers: np.ndarray  # (nx,ny,2)


@dataclass
class PlayerSnapshot:
    player_id: Optional[int]
    player_name: str
    team_name: str
    f_i: float
    v_eff: float
    a_eff: float
    d_eff: float
    tau_react_eff: float
    tau_turn_eff: float
    x: float
    y: float


@dataclass
class GameState:
    t_ms: int
    half_time: int
    ball_xy: Tuple[float,float]
    possession_team: Optional[str]
    carrier_id: Optional[int]  # if someone has the ball
    players: Dict[Tuple[Optional[int],str], PlayerSnapshot]
    last_event_row: Optional[pd.Series] = None
    # Lightweight caches
    control_field: Optional[np.ndarray] = None  # (nx,ny) team control advantage
    hazard_field: Optional[np.ndarray] = None   # (nx,ny) hazard 0..1


# =========================
# Fitting functions
# =========================

def _grid_centers(field: FieldConfig) -> np.ndarray:
    dx, dy = field.cell_size()
    xs = (np.arange(field.nx) + 0.5) * dx
    ys = (np.arange(field.ny) + 0.5) * dy
    g = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)  # (nx,ny,2)
    return g

def fit_ball_model(events: pd.DataFrame) -> BallModel:
    # Estimate kappa from passes if possible; else default 1.3
    df = events.copy()
    is_pass = df["event_type"].str.lower() == "pass"
    sub = df[is_pass].dropna(subset=["x_location_start","y_location_start","x_location_end","y_location_end"])
    if "event_end_time_in_ms" in sub.columns and "match_run_time_in_ms" in sub.columns:
        t0 = sub["match_run_time_in_ms"].astype("Int64")
        t1 = sub["event_end_time_in_ms"].astype("Int64")
        mask = t0.notna() & t1.notna() & (t1 > t0)
        sub = sub[mask].copy()
        if len(sub) >= 20:
            dx = (sub["x_location_end"] - sub["x_location_start"]).astype(float)
            dy = (sub["y_location_end"] - sub["y_location_start"]).astype(float)
            # Assume already in meters; if values look <=1.2, scale by (105,68)
            scale_x = 105.0 if dx.abs().max() <= 1.2 else 1.0
            scale_y = 68.0 if dy.abs().max() <= 1.2 else 1.0
            d = np.hypot(dx*scale_x, dy*scale_y)
            dt = (t1 - t0).astype(float) / 1000.0
            v0 = d / dt
            v0 = v0[(v0>0) & (v0<40)]  # rough guard
            if len(v0) >= 10:
                # In a pure exponential decay v(t)=v0*e^{-k t},
                # observed v0 spread should narrow with better k; we pick a k minimizing v0 variance
                ks = np.linspace(0.6, 2.0, 30)
                # Using implied "launch v0" from samples doesn't directly depend on k without acceleration model;
                # here we just pick a mid default (1.3) if distribution is sane.
                # TODO: replace with a better regression. For now return default unless outliers dominate.
                kappa = 1.3
                return BallModel(kappa_ground=float(kappa))
    return BallModel(kappa_ground=1.3)

def _zone_index(field: FieldConfig, x: float, y: float) -> Tuple[int,int]:
    ix = int(clamp(math.floor(x / (field.length_m/field.nx)), 0, field.nx-1))
    iy = int(clamp(math.floor(y / (field.width_m/field.ny)), 0, field.ny-1))
    return ix, iy

def fit_policy(events: pd.DataFrame, field: FieldConfig) -> PolicyModel:
    pm = PolicyModel()
    phase = lambda r: str(r.get("sequence_type","open")).lower()
    counts: Dict[Tuple[int,int,str], Counter] = defaultdict(Counter)
    for _, r in events.iterrows():
        try:
            xs, ys = float(r["x_location_start"]), float(r["y_location_start"])
        except Exception:
            continue
        ix, iy = _zone_index(field, xs, ys)
        a = str(r.get("event_type","")).lower()
        if a not in pm.actions:
            # map variants
            if a in {"pass","carry","dribble","take-on","takeon"}:
                a = "takeon" if a.startswith("take") else a
                if a == "dribble": a = "carry"
            elif a in {"shot","shoot"}: a = "shot"
            elif a in {"clearance","clear"}: a = "clear"
            else: continue
        counts[(ix,iy,phase(r))][a] += 1

    for key, c in counts.items():
        total = sum(c[a] for a in pm.actions)
        if total == 0:
            continue
        p = np.array([c[a]/total for a in pm.actions], dtype=float)
        # floor to avoid zeros
        p = (p + 1e-6); p = p / p.sum()
        pm.zone_phase_probs[key] = p
    return pm

def _bin_stats(vals: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    # returns bin centers and mean outcomes
    inds = np.digitize(vals, bins) - 1
    centers = 0.5*(bins[:-1] + bins[1:])
    means = np.zeros_like(centers)
    for i in range(len(centers)):
        mask = inds == i
        if mask.any():
            means[i] = np.mean(vals[mask])
        else:
            means[i] = np.nan
    return centers, means

def fit_success_models(events: pd.DataFrame) -> SuccessModel:
    sm = SuccessModel(params={})
    ev = events.copy()
    # Normalize flags
    ev["pressure_bin"] = ev["pressure"].astype(str).str.lower().isin({"1","true","t","yes","y"}).astype(int)
    ev["aerial_bin"] = (
        ev.get("body_type","").astype(str).str.contains("Header", case=False, na=False) |
        ev.get("style","").astype(str).str.contains("High|Cross|Chip|Long Ball", case=False, na=False) |
        ev.get("style_additional","").astype(str).str.contains("High|Cross|Chip|Long Ball", case=False, na=False)
    ).astype(int)

    # Geom: distance and cos(angle to goal simplified as receiver alignment)
    # For pass: use start->end vector length
    dx = (ev["x_location_end"] - ev["x_location_start"]).astype(float)
    dy = (ev["y_location_end"] - ev["y_location_start"]).astype(float)
    # scale if likely normalized
    sx = 105.0 if np.nanmax(np.abs(dx)) <= 1.2 else 1.0
    sy = 68.0 if np.nanmax(np.abs(dy)) <= 1.2 else 1.0
    d = np.hypot(dx*sx, dy*sy)
    ev["dist_m"] = d

    # Outcomes: rely on 'outcome' if exists; otherwise approximate: completed if end coords exist for pass/carry
    if "outcome" not in ev.columns:
        ev["outcome"] = np.where(ev["event_type"].str.lower().isin(["pass","carry","takeon"]),
                                 (~ev["x_location_end"].isna() & ~ev["y_location_end"].isna()).astype(int),
                                 np.nan)
    # Pass model
    pass_mask = ev["event_type"].str.lower()=="pass"
    sub = ev[pass_mask & ev["outcome"].isin([0,1])].copy()
    if len(sub) >= 30:
        # Simple logistic: logit p = b0 + b_d*(-d) + b_press*(-press) + b_aer*(aerial)
        # Initialize from binnings
        bins = np.linspace(0, 40, 9)
        centers, means = _bin_stats(sub["dist_m"].values, bins)
        # Fit monotone slope: difference between near and far bins
        if np.isfinite(means).sum() >= 3:
            near = np.nanmean(means[:2]); far = np.nanmean(means[-2:])
            b_d = (near - far) if np.isfinite(near) and np.isfinite(far) else 0.02
        else:
            b_d = 0.02
        p_base = sub["outcome"].mean()
        b0 = math.log(p_base/(1-p_base+1e-9)+1e-9)
        b_press = -0.3
        b_aer = -0.1
        sm.params["pass"] = dict(b0=b0, b_d=b_d, b_press=b_press, b_aer=b_aer)

    # Carry model
    carry_mask = ev["event_type"].str.lower()=="carry"
    sub = ev[carry_mask & ev["outcome"].isin([0,1])].copy()
    if len(sub) >= 20:
        bins = np.linspace(0, 25, 6)
        centers, means = _bin_stats(sub["dist_m"].values, bins)
        if np.isfinite(means).sum() >= 3:
            near = np.nanmean(means[:2]); far = np.nanmean(means[-2:])
            b_d = (near - far) if np.isfinite(near) and np.isfinite(far) else 0.02
        else:
            b_d = 0.02
        p_base = sub["outcome"].mean()
        b0 = math.log(p_base/(1-p_base+1e-9)+1e-9)
        b_press = -0.25
        sm.params["carry"] = dict(b0=b0, b_d=b_d, b_press=b_press)

    # Take-on model
    take_mask = ev["event_type"].str.lower().isin(["takeon","take-on"])
    sub = ev[take_mask & ev["outcome"].isin([0,1])].copy()
    if len(sub) >= 20:
        p_base = sub["outcome"].mean() if sub["outcome"].notna().any() else 0.45
        b0 = math.log(p_base/(1-p_base+1e-9)+1e-9)
        b_press = -0.3
        sm.params["takeon"] = dict(b0=b0, b_press=b_press)

    # Shot model (xG-lite)
    shot_mask = ev["event_type"].str.lower()=="shot"
    sub = ev[shot_mask].copy()
    if len(sub) >= 10:
        # if no outcome, set goal=1 else 0 as placeholder; in real feed use actual goal flag
        if "outcome" not in sub.columns or sub["outcome"].isna().all():
            sub["outcome"] = 0
        bins = np.linspace(5, 30, 6)
        centers, means = _bin_stats(sub["dist_m"].values, bins)
        slope = -0.08  # simple distance penalty
        p_base = max(0.02, min(0.2, np.nanmean(means) if np.isfinite(means).any() else 0.08))
        b0 = math.log(p_base/(1-p_base+1e-9)+1e-9)
        b_dist = slope
        sm.params["shot"] = dict(b0=b0, b_dist=b_dist)

    return sm

# =========================
# Control & hazard (lightweight)
# =========================

def control_field(field: FieldConfig, players: Dict[Tuple[Optional[int],str], 'PlayerSnapshot']) -> np.ndarray:
    # Very light proxy: for each grid cell, compute min TTR among players per team using v_eff & tau_react_eff
    g = _grid_centers(field)  # (nx,ny,2)
    teams = sorted({p.team_name for p in players.values() if p.team_name is not None})
    ttr = {team: np.full((field.nx, field.ny), fill_value=1e6, dtype=float) for team in teams}
    for p in players.values():
        if p.team_name is None: 
            continue
        # simple TTR: reaction delay + distance / max(v_eff, 0.1)
        v = max(0.1, float(p.v_eff))
        tau = max(0.01, float(p.tau_react_eff))
        dist = np.hypot(g[:,:,0]-p.x, g[:,:,1]-p.y)
        ttr_p = tau + dist / v
        ttr[p.team_name] = np.minimum(ttr[p.team_name], ttr_p)
    if len(teams) != 2:
        # return zeros if unknown
        return np.zeros((field.nx, field.ny), dtype=float)
    A, B = teams[0], teams[1]
    # control advantage: negative means B faster, positive means A faster
    adv = ttr[A] - ttr[B]
    # squash to [-1,1] by scaling
    adv = np.tanh(adv / 0.8)
    return adv

def hazard_field(field: FieldConfig, players: Dict[Tuple[Optional[int],str], 'PlayerSnapshot'], pressure_prior=0.25) -> np.ndarray:
    # Simple hazard: inverse of distance to nearest opponent + prior
    g = _grid_centers(field)
    # assume first half of players belong to team A, second to team B - we just compute crowding
    positions = np.array([[p.x, p.y] for p in players.values()])
    if len(positions) == 0:
        return np.full((field.nx, field.ny), pressure_prior, dtype=float)
    # nearest distance to any player
    dmin = np.full((field.nx, field.ny), 50.0, dtype=float)
    for p in players.values():
        dist = np.hypot(g[:,:,0]-p.x, g[:,:,1]-p.y)
        dmin = np.minimum(dmin, dist)
    haz = 1.0 / (1.0 + dmin)  # 0..1-ish
    haz = np.clip(haz + pressure_prior, 0.0, 1.0)
    return haz


# =========================
# Success scoring
# =========================

def _success_prob(action: str, features: Dict[str,float], sm: SuccessModel) -> float:
    p = 0.5
    if action not in sm.params:
        return p
    coefs = sm.params[action]
    z = coefs.get("b0", 0.0)
    if action == "pass":
        z += coefs.get("b_d", 0.02) * (-features.get("dist_m",0.0))
        z += coefs.get("b_press", -0.3) * (-features.get("pressure",0.0))
        z += coefs.get("b_aer", -0.1) * (features.get("aerial",0.0))
    elif action == "carry":
        z += coefs.get("b_d", 0.02) * (-features.get("dist_m",0.0))
        z += coefs.get("b_press", -0.25) * (-features.get("pressure",0.0))
    elif action == "takeon":
        z += coefs.get("b_press", -0.3) * (-features.get("pressure",0.0))
    elif action == "shot":
        z += coefs.get("b_dist", -0.08) * (features.get("dist_m",0.0))
    p = logistic(z)
    return float(clamp(p, sm.p_min, sm.p_max))


# =========================
# Policy sampling
# =========================

def _phase(row: Optional[pd.Series]) -> str:
    return "open" if row is None else str(row.get("sequence_type","open")).lower()

def sample_action_probs(ix:int, iy:int, phase:str, policy: PolicyModel) -> Tuple[List[str], np.ndarray]:
    key = (ix,iy,phase)
    if key in policy.zone_phase_probs:
        p = policy.zone_phase_probs[key]
        return policy.actions, p
    # fallback: uniform
    p = np.ones(len(policy.actions), dtype=float) / len(policy.actions)
    return policy.actions, p

def sample_action(ix:int, iy:int, phase:str, policy: PolicyModel, rng: np.random.Generator) -> str:
    actions, p = sample_action_probs(ix, iy, phase, policy)
    a = rng.choice(actions, p=p)
    return str(a)


# =========================
# Simulator
# =========================

@dataclass
class SimConfig:
    horizon_K: int = 5
    rollouts_R: int = 200
    half_life_events: float = 3.0  # confidence halves every H events
    topM: int = 7
    seed: Optional[int] = 7

def _confidence_step(p_succ: float, state_sharp: float, data_support: float) -> float:
    # model confidence (peak prob)
    c_model = max(p_succ, 1.0 - p_succ)
    c_state = float(clamp(state_sharp, 0.0, 1.0))
    c_data  = float(clamp(data_support, 0.0, 1.0))
    return geometric_mean([c_model, c_state, c_data])

def _state_sharpness(ctrl_adv_here: float) -> float:
    # |advantage| near 1 => very sharp; near 0 => ambiguous
    return float(abs(ctrl_adv_here))

def _data_support(action: str, ix: int, iy: int, phase: str, policy: PolicyModel) -> float:
    key = (ix,iy,phase)
    if key in policy.zone_phase_probs:
        # more peaked distributions imply more data or stronger signal; use entropy
        p = policy.zone_phase_probs[key]
        H = entropy(p)
        Hmax = math.log(len(p))
        return float(1.0 - H/Hmax)
    return 0.2  # weak default

def _action_geometry(action: str, carrier: PlayerSnapshot, field: FieldConfig, rng: np.random.Generator) -> Dict[str,float]:
    # Very simple geometry sampler
    if action == "pass":
        # random target within 20m with slight forward bias
        r = rng.uniform(6, 22)
        theta = rng.normal(loc=0.0, scale=np.pi/6)  # forward-ish
        dx, dy = r*np.cos(theta), r*np.sin(theta)
        tx, ty = clamp(carrier.x+dx, 0, field.length_m), clamp(carrier.y+dy, 0, field.width_m)
        dist = np.hypot(tx-carrier.x, ty-carrier.y)
        return {"tx":tx,"ty":ty,"dist_m":dist,"aerial":0.0}
    if action == "carry":
        r = rng.uniform(3, 12)
        theta = rng.normal(loc=0.0, scale=np.pi/8)
        tx, ty = clamp(carrier.x+r*np.cos(theta), 0, field.length_m), clamp(carrier.y+r*np.sin(theta), 0, field.width_m)
        dist = np.hypot(tx-carrier.x, ty-carrier.y)
        return {"tx":tx,"ty":ty,"dist_m":dist}
    if action == "takeon":
        r = rng.uniform(2, 6)
        theta = rng.normal(loc=0.0, scale=np.pi/5)
        tx, ty = clamp(carrier.x+r*np.cos(theta), 0, field.length_m), clamp(carrier.y+r*np.sin(theta), 0, field.width_m)
        dist = np.hypot(tx-carrier.x, ty-carrier.y)
        return {"tx":tx,"ty":ty,"dist_m":dist}
    if action == "shot":
        # aim at goal center (right side goal at x=105, y=34)
        tx, ty = field.length_m, field.width_m/2
        dist = np.hypot(tx-carrier.x, ty-carrier.y)
        return {"tx":tx,"ty":ty,"dist_m":dist}
    if action == "clear":
        tx, ty = carrier.x, clamp(carrier.y + (rng.choice([-1,1])*rng.uniform(8,20)), 0, field.width_m)
        dist = abs(ty-carrier.y)
        return {"tx":tx,"ty":ty,"dist_m":dist}
    return {"tx":carrier.x,"ty":carrier.y,"dist_m":0.0}

def _ixiy(field: FieldConfig, x: float, y: float) -> Tuple[int,int]:
    ix = int(clamp(int(x / (field.length_m/field.nx)), 0, field.nx-1))
    iy = int(clamp(int(y / (field.width_m/field.ny)), 0, field.ny-1))
    return ix, iy

def simulate_rollouts(initial_state: GameState, fit: MatchFit, cfg: SimConfig) -> Dict[str,Any]:
    rng = np.random.default_rng(cfg.seed)
    field = fit.field
    lambda_ = math.log(2.0) / max(1e-6, cfg.half_life_events)

    # ensure control/hazard for initial state
    if initial_state.control_field is None:
        initial_state.control_field = control_field(field, initial_state.players)
    if initial_state.hazard_field is None:
        initial_state.hazard_field = hazard_field(field, initial_state.players)

    results = []
    actions = fit.policy.actions

    for r in range(cfg.rollouts_R):
        s = initial_state  # Shallow copy is OK for now since we mutate primitives only
        seq, confs = [], []
        t_ms = s.t_ms
        carrier_key = (s.carrier_id, next((p.player_name for k,p in s.players.items() if k[0]==s.carrier_id), "")) if s.carrier_id is not None else None
        if carrier_key is None or carrier_key not in s.players:
            # pick nearest to ball as carrier
            min_d, min_k = 1e9, None
            bx, by = s.ball_xy
            for k, p in s.players.items():
                d = math.hypot(p.x-bx, p.y-by)
                if d < min_d:
                    min_d, min_k = d, k
            carrier_key = min_k

        for k in range(1, cfg.horizon_K+1):
            carrier = s.players[carrier_key]
            ix, iy = _ixiy(field, carrier.x, carrier.y)
            phase = _phase(s.last_event_row)

            # sample action from policy
            action = sample_action(ix, iy, phase, fit.policy, rng)

            # sample geometry
            geom = _action_geometry(action, carrier, field, rng)

            # pressure proxy from local hazard
            ctrl_adv = s.control_field[ix,iy] if s.control_field is not None else 0.0
            hazard_here = s.hazard_field[ix,iy] if s.hazard_field is not None else 0.25
            pressure = float(hazard_here > 0.5)

            # success prob
            feats = {"dist_m": geom["dist_m"], "pressure": pressure, "aerial": float(action=="pass" and rng.random()<0.15)}
            p_succ = _success_prob(action, feats, fit.success)

            # draw outcome
            success = (rng.random() < p_succ)

            # step confidence
            c_step = _confidence_step(p_succ, _state_sharpness(ctrl_adv), _data_support(action, ix, iy, phase, fit.policy))
            confs.append(c_step)

            # advance simple state: move ball (and carrier) to target on success, perturb on failure
            tx, ty = geom["tx"], geom["ty"]
            if success:
                # if pass, ball goes to target; select new carrier as nearest same-team player
                if action == "pass":
                    # set ball
                    s.ball_xy = (tx, ty)
                    # pick teammate nearest to target as new carrier
                    team = carrier.team_name
                    nearest, best_key = 1e9, carrier_key
                    for k2, p2 in s.players.items():
                        if p2.team_name != team: continue
                        d2 = math.hypot(p2.x - tx, p2.y - ty)
                        if d2 < nearest:
                            nearest, best_key = d2, k2
                    carrier_key = best_key
                elif action in ("carry","takeon","clear"):
                    # carrier moves to target
                    carrier.x, carrier.y = tx, ty
                    s.ball_xy = (tx, ty)
                elif action == "shot":
                    s.ball_xy = (tx, ty)  # near goal
                s.possession_team = carrier.team_name
            else:
                # turnover probability on fail
                turnover = (action in ("pass","takeon","carry")) and (rng.random() < 0.7)
                if turnover:
                    # possession flips to nearest opponent
                    opp_team = None
                    for p in s.players.values():
                        if p.team_name != carrier.team_name:
                            opp_team = p.team_name; break
                    s.possession_team = opp_team
                    s.ball_xy = (tx, ty)
                    # choose opponent nearest to ball as new carrier
                    nearest, best_key = 1e9, carrier_key
                    for k2, p2 in s.players.items():
                        if p2.team_name != opp_team: continue
                        d2 = math.hypot(p2.x - tx, p2.y - ty)
                        if d2 < nearest:
                            nearest, best_key = d2, k2
                    carrier_key = best_key
                else:
                    # retain but perturb position slightly
                    jitter = 2.0 * (rng.random() - 0.5)
                    carrier.x = clamp(carrier.x + jitter, 0, field.length_m)
                    carrier.y = clamp(carrier.y + jitter, 0, field.width_m)
                    s.ball_xy = (carrier.x, carrier.y)

            # simple time advance: travel time + control delay
            travel_v = max(12.0, 18.0 * np.exp(-fit.ball.kappa_ground*0.2))  # crude proxy
            dt_travel = geom["dist_m"] / travel_v
            dt_ctrl = fit.ball.control_delay(under_pressure=bool(pressure))
            t_ms += int(1000 * (dt_travel + dt_ctrl))

            # refresh fields sparsely (every step)
            s.control_field = control_field(fit.field, s.players)
            s.hazard_field = hazard_field(fit.field, s.players)

            seq.append({
                "k": k,
                "actor_id": carrier.player_id,
                "actor_name": carrier.player_name,
                "team": carrier.team_name,
                "action": action,
                "success": int(success),
                "p_succ": float(p_succ),
                "tx": float(tx), "ty": float(ty),
                "dist_m": float(geom["dist_m"]),
                "t_ms": int(t_ms),
                "ctrl_adv": float(ctrl_adv),
                "hazard_here": float(hazard_here),
                "c_step": float(c_step),
            })

        # sequence confidence with horizon decay
        C_seq = float(np.prod(confs) * math.exp(-lambda_ * len(confs)))
        results.append({"seq": seq, "C_seq": C_seq})

    # Aggregate: group sequences by coarse action-type pattern
    pattern_counts = Counter()
    for r in results:
        pat = tuple(e["action"] for e in r["seq"])
        pattern_counts[pat] += 1
    top_patterns = pattern_counts.most_common()

    # Summaries
    mean_conf = float(np.mean([r["C_seq"] for r in results])) if results else 0.0
    shot_by_k = Counter()
    turnover_by_k = Counter()
    for r in results:
        seq = r["seq"]
        for e in seq:
            if e["action"] == "shot":
                shot_by_k[e["k"]] += 1
        # crude: turnover == success==0 & possession flipped is not tracked, skip

    out = {
        "mean_sequence_confidence": mean_conf,
        "top_patterns": [(list(p), int(c)) for p,c in top_patterns[:10]],
        "shot_prob_by_k": {int(k): (shot_by_k[k]/cfg.rollouts_R) for k in shot_by_k},
        "results": results[: min(50, len(results))]  # cap for readability
    }
    return out


# =========================
# Fitting entry point
# =========================

def fit_match_model(merged_events: pd.DataFrame) -> MatchFit:
    # Basic checks
    for c in REQ_COLS:
        if c not in merged_events.columns:
            raise ValueError(f"Missing required column: {c}")

    # Normalize coords to meters if they look normalized
    df = merged_events.copy()
    # Use extrema to decide
    max_abs = np.nanmax(np.abs(df[["x_location_start","x_location_end","y_location_start","y_location_end"]].to_numpy(dtype=float)))
    if max_abs <= 1.2:
        # scale to 105x68
        for c in ["x_location_start","x_location_end"]:
            df[c] = df[c].astype(float) * 105.0
        for c in ["y_location_start","y_location_end"]:
            df[c] = df[c].astype(float) * 68.0

    field = FieldConfig()
    grid = _grid_centers(field)
    ball = fit_ball_model(df)
    policy = fit_policy(df, field)
    success = fit_success_models(df)

    return MatchFit(field=field, ball=ball, policy=policy, success=success, grid_centers=grid)


# =========================
# State construction
# =========================

def construct_state_from_row(row: pd.Series, team_rosters: Dict[str, List[Tuple[Optional[int],str, float,float]]]) -> GameState:
    """
    Build a GameState around a given event row, given approximate player positions for both teams.
    team_rosters: dict team_name -> list of tuples (player_id, player_name, x, y)
    """
    players = {}
    for team, roster in team_rosters.items():
        for pid, pname, x, y in roster:
            players[(pid, pname)] = PlayerSnapshot(
                player_id=pid, player_name=pname, team_name=team,
                f_i=float(row.get("syn_f_i", 1.0)),
                v_eff=float(row.get("v_i_eff", 6.0)),
                a_eff=float(row.get("a_i_eff", 3.0)),
                d_eff=float(row.get("d_i_eff", 4.0)),
                tau_react_eff=float(row.get("tau_i_react_eff", 0.25)),
                tau_turn_eff=float(row.get("tau_i_turn_eff", 0.34)),
                x=float(x), y=float(y)
            )
    bx = float(row.get("x_location_start", 52.5))
    by = float(row.get("y_location_start", 34.0))
    t_ms = int(row.get("match_run_time_in_ms", row.get("event_end_time_in_ms", 0)) or 0)
    half_time = int(row.get("half_time", 1) or 1)
    carrier_id = int(row.get("from_player_id")) if not pd.isna(row.get("from_player_id")) else None
    gs = GameState(
        t_ms=t_ms, half_time=half_time, ball_xy=(bx,by),
        possession_team=str(row.get("team_name","")) or None,
        carrier_id=carrier_id, players=players, last_event_row=row
    )
    # lazy compute fields on first use
    return gs
