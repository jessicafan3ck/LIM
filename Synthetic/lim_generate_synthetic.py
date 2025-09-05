
import os
import re
import glob
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from collections import deque, defaultdict

import numpy as np
import pandas as pd

@dataclass
class PhysParams:
    v_max_ms: float
    a_max: float
    d_max: float
    tau_react: float
    tau_turn: float
    is_gk: bool = False
    tau_gk_react: Optional[float] = None
    r_gk: Optional[float] = None
    v_dive: Optional[float] = None

@dataclass
class FatigueState:
    F_fast: float = 1.0
    F_slow: float = 1.0
    last_time_ms: Optional[int] = None
    tau_mult_until_ms: int = 0
    recent_bursts: deque = field(default_factory=lambda: deque())

DEFAULT_V_MAX_MS = 8.6
MAX_A_CAP = 6.5
MAX_D_CAP = 8.0

TAU_FAST_BASE = 45.0
TAU_SLOW_BASE = 600.0

ALPHA_BURST = 0.20
ALPHA_CONTACT = 0.30
BETA_BURST = 0.10
BETA_CONTACT = 0.20

HEAVY_CONTACT_THRESHOLD = 0.7
HEAVY_CONTACT_TAU_MULT = 1.5
HEAVY_CONTACT_WINDOW_S = 60.0

F_BLEND_FAST = 0.6
F_BLEND_SLOW = 0.4

K_V = 0.25
K_A = 0.40
K_D = 0.20
K_TAU = 0.50

GK_R = 3.0
GK_V_DIVE = 6.0

COLS = {
    "match_id": "match_id",
    "event_id": "event_id",
    "time_start_ms": "match_run_time_in_ms",
    "time_end_ms": "event_end_time_in_ms",
    "event": "event",
    "event_type": "event_type",
    "from_pid": "from_player_id",
    "from_pname": "from_player_name",
    "to_pname": "to_player_name",
    "team": "team_name",
    "x_start": "x_location_start",
    "y_start": "y_location_start",
    "x_end": "x_location_end",
    "y_end": "y_location_end",
    "pressure": "pressure",
    "body_type": "body_type",
    "style": "style",
    "style_add": "style_additional",
    "jersey": "from_player_shirt_number",
}

PHYS_COLS = {
    "match_id": "Match ID",
    "player_id": "Player ID",
    "player_name": "Player Name",
    "team_name": "Team Name",
    "max_speed": "Max Speed (km/h)",
    "n_sprints": "# Sprints",
    "n_speed_runs": "# Speed Runs",
    "dur_min": "Total Duration (min)",
    "jersey": "Jersey #",
}

def kmh_to_ms(v_kmh: float) -> float:
    if pd.isna(v_kmh):
        return np.nan
    return float(v_kmh) / 3.6

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def calc_t85(n_sprints: Optional[float]) -> float:
    n = 0.0 if pd.isna(n_sprints) else float(n_sprints)
    return clamp(2.2 - 0.01 * n, 1.6, 2.2)

def derive_phys_params(phys_df: pd.DataFrame):
    params = {}
    v_by_team = {}
    for team, sub in phys_df.groupby(PHYS_COLS["team_name"]):
        valid = sub[PHYS_COLS["max_speed"]].dropna().astype(float)
        v_by_team[team] = kmh_to_ms(valid.median()) if len(valid) > 0 else DEFAULT_V_MAX_MS

    for _, r in phys_df.iterrows():
        pid = r.get(PHYS_COLS["player_id"], np.nan)
        pname = r.get(PHYS_COLS["player_name"], None)
        team = r.get(PHYS_COLS["team_name"], None)
        n_sprints = r.get(PHYS_COLS["n_sprints"], np.nan)
        v_kmh = r.get(PHYS_COLS["max_speed"], np.nan)

        v_max = kmh_to_ms(v_kmh) if not pd.isna(v_kmh) else v_by_team.get(team, DEFAULT_V_MAX_MS)
        t85 = calc_t85(n_sprints)
        a_max = 0.85 * v_max / t85
        d_max = 1.25 * a_max
        a_max = min(a_max, MAX_A_CAP)
        d_max = min(d_max, MAX_D_CAP)

        n = 0.0 if pd.isna(n_sprints) else float(n_sprints)
        tau_react = clamp(0.26 - 0.002 * min(20.0, n), 0.18, 0.30)
        tau_turn = clamp(0.34 - 0.002 * min(20.0, n), 0.26, 0.40)

        params[(int(pid) if not pd.isna(pid) else None, str(pname) if pname is not None else "")] = dict(
            v_max_ms=v_max, a_max=a_max, d_max=d_max, tau_react=tau_react, tau_turn=tau_turn, is_gk=False,
            tau_gk_react=None, r_gk=None, v_dive=None
        )
    return params

CONTACT_RE = re.compile(r"(foul|tackle|aerial|duel|collision|challenge)", re.IGNORECASE)
SAVE_RE = re.compile(r"(save|goal\s?kick|keeper|gk)", re.IGNORECASE)

def detect_contact(row: pd.Series) -> float:
    text = " ".join(str(row.get(col, "")) for col in ["event", "event_type", "body_type"]
                    if col in row and pd.notna(row[col]))
    if not text.strip():
        return 0.0
    if CONTACT_RE.search(text):
        if re.search(r"aerial|header", text, re.IGNORECASE):
            return 0.4
        if re.search(r"foul|collision", text, re.IGNORECASE):
            return 0.7
        return 0.4
    return 0.0

def carry_displacement_intensity(row: pd.Series) -> float:
    try:
        if pd.notna(row.get("event_type")) and str(row["event_type"]).lower() == "carry":
            xs, ys, xe, ye = (row.get("x_location_start"), row.get("y_location_start"),
                              row.get("x_location_end"), row.get("y_location_end"))
            if all(pd.notna(v) for v in [xs, ys, xe, ye]):
                scale_x = 105.0 if max(abs(float(xs)), abs(float(xe))) <= 1.2 else 1.0
                scale_y = 68.0 if max(abs(float(ys)), abs(float(ye))) <= 1.2 else 1.0
                dx = (float(xe) - float(xs)) * scale_x
                dy = (float(ye) - float(ys)) * scale_y
                d = float(np.hypot(dx, dy))
                if d >= 12:
                    return 1.0
                elif d >= 8:
                    return 0.6
                elif d >= 4:
                    return 0.3
    except Exception:
        pass
    return 0.0

def choose_time_ms(row: pd.Series):
    for c in ["event_end_time_in_ms", "match_run_time_in_ms", "match_time_in_ms"]:
        if c in row and pd.notna(row[c]):
            try:
                return int(row[c])
            except Exception:
                continue
    return None

def detect_gks_from_events(events: pd.DataFrame):
    counts = defaultdict(int)
    gk_marks = defaultdict(bool)
    for _, r in events.iterrows():
        pid = r.get("from_player_id", np.nan)
        pname = r.get("from_player_name", None)
        key = (int(pid) if not pd.isna(pid) else None, str(pname) if pname is not None else "")
        text = " ".join(str(r.get(col, "")) for col in ["event", "event_type"] if col in r and pd.notna(r[col]))
        if SAVE_RE.search(text):
            gk_marks[key] = True
        counts[key] += 1
    if not any(gk_marks.values()):
        try:
            jersey_df = events[["from_player_id", "from_player_name", "from_player_shirt_number"]].dropna()
            jersey_df["from_player_shirt_number"] = jersey_df["from_player_shirt_number"].astype(str)
            mask = jersey_df["from_player_shirt_number"].isin({"1", "13"})
            if mask.any():
                jr = jersey_df[mask].iloc[0]
                key = (int(jr["from_player_id"]) if not pd.isna(jr["from_player_id"]) else None,
                       str(jr["from_player_name"]) if pd.notna(jr["from_player_name"]) else "")
                gk_marks[key] = True
        except Exception:
            pass
    return gk_marks

def generate_synthetic_sidecar(events: pd.DataFrame, phys_df_concat: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    phys_params_map = {}
    if phys_df_concat is not None and len(phys_df_concat) > 0:
        phys_params_map = derive_phys_params(phys_df_concat)

    gk_map = detect_gks_from_events(events)

    team_v_defaults = {}
    if phys_df_concat is not None and len(phys_df_concat) > 0:
        for team, sub in phys_df_concat.groupby(PHYS_COLS["team_name"]):
            vals = sub[PHYS_COLS["max_speed"]].dropna().astype(float)
            team_v_defaults[team] = kmh_to_ms(vals.median()) if len(vals) > 0 else DEFAULT_V_MAX_MS

    states = defaultdict(lambda: dict(F_fast=1.0, F_slow=1.0, last_time_ms=None, tau_mult_until_ms=0, recent_bursts=[]))
    out_rows = []

    events_sorted = events.copy()
    events_sorted["__time_ms"] = events_sorted.apply(choose_time_ms, axis=1)
    events_sorted = events_sorted.sort_values(["match_id", "__time_ms", "event_id"], kind="mergesort")

    last_event_time = defaultdict(lambda: None)

    for _, row in events_sorted.iterrows():
        mid = row.get("match_id", np.nan)
        eid = row.get("event_id", np.nan)
        pid = row.get("from_player_id", np.nan)
        pname = row.get("from_player_name", None)
        team = row.get("team_name", None)

        key = (int(pid) if not pd.isna(pid) else None, str(pname) if pname is not None else "")

        pp = phys_params_map.get(key)
        if pp is None:
            v_max = team_v_defaults.get(team, DEFAULT_V_MAX_MS)
            t85 = max(1.6, min(2.2, 2.2 - 0.01 * 8))
            a_max = min(0.85 * v_max / t85, MAX_A_CAP)
            d_max = min(1.25 * a_max, MAX_D_CAP)
            tau_react = max(0.18, min(0.30, 0.26 - 0.002 * 8))
            tau_turn = max(0.26, min(0.40, 0.34 - 0.002 * 8))
            pp = dict(v_max_ms=v_max, a_max=a_max, d_max=d_max, tau_react=tau_react, tau_turn=tau_turn,
                      is_gk=False, tau_gk_react=None, r_gk=None, v_dive=None)

        is_gk = gk_map.get(key, False)
        if is_gk and not pp["is_gk"]:
            pp["is_gk"] = True
            pp["tau_gk_react"] = pp["tau_react"] + 0.02
            pp["r_gk"] = GK_R
            pp["v_dive"] = GK_V_DIVE

        t_ms = row["__time_ms"]
        st = states[key]
        if st["last_time_ms"] is None:
            st["last_time_ms"] = t_ms

        dt_s = 0.0
        if t_ms is not None and st["last_time_ms"] is not None:
            dt_s = max(0.0, (int(t_ms) - int(st["last_time_ms"])) / 1000.0)

        tau_mult = 1.0
        if st["tau_mult_until_ms"] and t_ms is not None and int(t_ms) < int(st["tau_mult_until_ms"]):
            tau_mult = HEAVY_CONTACT_TAU_MULT
        tau_fast = TAU_FAST_BASE * tau_mult
        tau_slow = TAU_SLOW_BASE * tau_mult

        if dt_s > 0:
            st["F_fast"] = 1.0 - (1.0 - st["F_fast"]) * np.exp(-dt_s / tau_fast)
            st["F_slow"] = 1.0 - (1.0 - st["F_slow"]) * np.exp(-dt_s / tau_slow)

        I_burst = carry_displacement_intensity(row)
        pres = str(row.get("pressure", "")).strip().lower() in {"1", "true", "yes", "y", "t"}
        if pres:
            I_burst = max(I_burst, 0.4)

        last_t = last_event_time.get(key)
        quick_chain = 0
        if last_t is not None and t_ms is not None and (int(t_ms) - int(last_t)) <= 5000:
            I_burst += 0.3
            quick_chain = 1

        if t_ms is not None:
            st["recent_bursts"].append((int(t_ms), I_burst))
            st["recent_bursts"] = [(tm, ib) for (tm, ib) in st["recent_bursts"] if int(t_ms) - tm <= 30000]
            sum_bursts = sum(v for _, v in st["recent_bursts"])
            I_burst_long = max(0.0, min(1.0, sum_bursts / 30.0))
        else:
            I_burst_long = 0.0

        C = detect_contact(row)

        st["F_fast"] = float(np.clip(st["F_fast"] - (0.20 * I_burst + 0.30 * C), 0.0, 1.0))
        st["F_slow"] = float(np.clip(st["F_slow"] - (0.10 * I_burst_long + 0.20 * C), 0.0, 1.0))

        if C >= 0.7 and t_ms is not None:
            st["tau_mult_until_ms"] = int(t_ms) + 60000

        f_i = 0.6 * st["F_fast"] + 0.4 * st["F_slow"]

        v_eff = pp["v_max_ms"] * (1.0 - 0.25 * (1.0 - f_i))
        a_eff = pp["a_max"] * (1.0 - 0.40 * (1.0 - f_i))
        d_eff = pp["d_max"] * (1.0 - 0.20 * (1.0 - f_i))
        tau_react_eff = pp["tau_react"] * (1.0 + 0.50 * (1.0 - f_i))
        tau_turn_eff = pp["tau_turn"] * (1.0 + 0.50 * (1.0 - f_i))
        if pp["is_gk"] and pp.get("tau_gk_react") is not None:
            tau_react_eff = pp["tau_gk_react"] * (1.0 + 0.50 * (1.0 - f_i))

        out_rows.append({
            "match_id": mid,
            "event_id": eid,
            "from_player_id": key[0],
            "from_player_name": key[1],
            "team_name": team,
            "syn_F_fast": st["F_fast"],
            "syn_F_slow": st["F_slow"],
            "syn_f_i": f_i,
            "syn_I_burst": I_burst,
            "syn_I_burst_long": I_burst_long,
            "syn_contact": C,
            "syn_quick_chain": quick_chain,
            "v_i_max": pp["v_max_ms"],
            "a_i_max": pp["a_max"],
            "d_i_max": pp["d_max"],
            "tau_i_react": pp["tau_react"],
            "tau_i_turn": pp["tau_turn"],
            "is_gk": int(pp["is_gk"]),
            "tau_GK_react": pp.get("tau_gk_react", np.nan) if pp["is_gk"] else np.nan,
            "r_GK": pp.get("r_gk", np.nan) if pp["is_gk"] else np.nan,
            "v_dive": pp.get("v_dive", np.nan) if pp["is_gk"] else np.nan,
            "v_i_eff": v_eff,
            "a_i_eff": a_eff,
            "d_i_eff": d_eff,
            "tau_i_react_eff": tau_react_eff,
            "tau_i_turn_eff": tau_turn_eff,
        })

        if t_ms is not None:
            last_event_time[key] = int(t_ms)
            st["last_time_ms"] = int(t_ms)

    sidecar = pd.DataFrame(out_rows)
    float_cols = [c for c in sidecar.columns if c.startswith("syn_") or c.endswith("_eff") or c.endswith("_max") or c.startswith("tau_")]
    sidecar[float_cols] = sidecar[float_cols].astype(float)
    return sidecar

def load_all_phys(phys_dir: str):
    if not os.path.isdir(phys_dir):
        return None
    files = sorted(glob.glob(os.path.join(phys_dir, "*.xlsx")) + glob.glob(os.path.join(phys_dir, "*.xls")))
    frames = []
    for p in files:
        try:
            df = pd.read_excel(p, sheet_name=0)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipping phys file {p}: {e}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def run_batch(base_dir: str = ".", events_subdir="Events", phys_subdir="Phys_Summary", synthetic_subdir="Synthetic"):
    events_dir = os.path.join(base_dir, events_subdir)
    phys_dir = os.path.join(base_dir, phys_subdir)
    out_dir = os.path.join(base_dir, synthetic_subdir)
    os.makedirs(out_dir, exist_ok=True)

    phys_df_concat = load_all_phys(phys_dir)

    event_paths = sorted(glob.glob(os.path.join(events_dir, "*.xlsx")) + glob.glob(os.path.join(events_dir, "*.xls")))
    if not event_paths:
        print(f"[INFO] No Events files found in {events_dir}")
        return

    for ev_path in event_paths:
        try:
            ev = pd.read_excel(ev_path, sheet_name=0)
        except Exception as e:
            print(f"[WARN] Skipping events file {ev_path}: {e}")
            continue

        sidecar = generate_synthetic_sidecar(ev, phys_df_concat)
        base = os.path.splitext(os.path.basename(ev_path))[0]
        out_csv = os.path.join(out_dir, f"{base}_synthetic.csv")
        sidecar.to_csv(out_csv, index=False)
        print(f"[OK] Wrote {out_csv}")

if __name__ == "__main__":
    run_batch(base_dir="/mnt/data")
