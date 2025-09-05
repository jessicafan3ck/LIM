# high-level goal

Given:

* one or more **Events** spreadsheets (FIFA unified event data), and
* optionally one or more **Physical Summary** spreadsheets (per-player match physicals),

the script builds, **for each events file**, a **synthetic sidecar CSV** with per-event, per-player dynamic state needed by LIM:

* instantaneous **fatigue reservoirs** (`syn_F_fast`, `syn_F_slow`) and their blend `syn_f_i`
* **burst/contact** intensities driving fatigue (`syn_I_burst`, `syn_I_burst_long`, `syn_contact`, `syn_quick_chain`)
* **base kinematic caps** from physicals (`v_i_max`, `a_i_max`, `d_i_max`, `tau_i_react`, `tau_i_turn`, GK extras)
* **effective kinematics** after fatigue (`v_i_eff`, `a_i_eff`, `d_i_eff`, `tau_i_react_eff`, `tau_i_turn_eff`)

All outputs keep keys (`match_id`, `event_id`, `from_player_id`, `from_player_name`, `team_name`) so you can left-join back to the events table.

---

# imports & types

```python
import os, re, glob
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from collections import deque, defaultdict
import numpy as np
import pandas as pd
```

* `os`, `glob`: filesystem walking and path building.
* `re`: regex for tagging contact events and GK cues.
* `dataclasses`: `@dataclass` convenience containers for per-player parameters and state.
* `typing`: precise keys for dicts.
* `deque`, `defaultdict`: efficient rolling windows (bursts) & auto-initializing maps.
* `numpy`, `pandas`: vector math, I/O, frames.

---

# dataclasses (state containers)

## `PhysParams`

```python
@dataclass
class PhysParams:
    v_max_ms: float        # top speed cap (m/s)
    a_max: float           # max acceleration cap (m/s^2)
    d_max: float           # max deceleration/braking cap (m/s^2)
    tau_react: float       # reaction latency base (s)
    tau_turn: float        # turning latency base (s)
    is_gk: bool = False    # goalkeeper flag
    tau_gk_react: Optional[float] = None # GK reaction base (s)
    r_gk: Optional[float] = None         # GK dive reach radius (m)
    v_dive: Optional[float] = None       # GK dive speed (m/s)
```

* One instance per player **identity** (ID+name). Values mostly come from physical summaries; if missing, we synthesize (details later).

## `FatigueState`

```python
@dataclass
class FatigueState:
    F_fast: float = 1.0               # fast reservoir (0..1), recovers ~45s
    F_slow: float = 1.0               # slow reservoir (0..1), recovers ~600s
    last_time_ms: Optional[int] = None
    tau_mult_until_ms: int = 0        # heavy contact slows recovery until this time (ms)
    recent_bursts: deque = field(default_factory=lambda: deque())  # (time_ms, I_burst)
```

* One instance **per player, per match timeline**.
* Tracks recovery between events, applies decrements on events, and carries a 30s rolling burst window.

---

# model constants (design choices)

```python
DEFAULT_V_MAX_MS = 8.6
MAX_A_CAP = 6.5
MAX_D_CAP = 8.0
```

* Fallback top speed if no physicals: **8.6 m/s** (\~31 km/h). Reasonable for youth elite top-end.
* Physical caps: acceleration **≤ 6.5 m/s²**, braking **≤ 8.0 m/s²** (braking > acceleration).

```python
TAU_FAST_BASE = 45.0     # seconds (fast reservoir)
TAU_SLOW_BASE = 600.0    # seconds (slow reservoir)
```

* Exponential recovery time constants: quick cardiovascular/neuromuscular rebound vs slower metabolic fatigue.

```python
ALPHA_BURST = 0.20
ALPHA_CONTACT = 0.30
BETA_BURST = 0.10
BETA_CONTACT = 0.20
```

* **Event-time decrements** to reservoirs:

  * `F_fast` drops by `ALPHA_BURST*I_burst + ALPHA_CONTACT*C`
  * `F_slow` drops by `BETA_BURST*I_burst_long + BETA_CONTACT*C`
* Heavier weight on **contact**; fast reservoir more sensitive to **acute bursts**, slow reservoir to **accumulation** (`I_burst_long`).

```python
HEAVY_CONTACT_THRESHOLD = 0.7
HEAVY_CONTACT_TAU_MULT = 1.5
HEAVY_CONTACT_WINDOW_S = 60.0
```

* If contact intensity `C ≥ 0.7` (e.g., foul/collision), for **60s** the recovery taus are multiplied by **1.5**, i.e., recovery slows.

```python
F_BLEND_FAST = 0.6
F_BLEND_SLOW = 0.4
```

* Final fatigue multiplier `f_i = 0.6*F_fast + 0.4*F_slow`.

```python
K_V = 0.25
K_A = 0.40
K_D = 0.20
K_TAU = 0.50
```

* Effective kinematics are scaled by fatigue:

  * `v_i_eff = v_max * (1 - 0.25*(1 - f))`
  * `a_i_eff = a_max * (1 - 0.40*(1 - f))`
  * `d_i_eff = d_max * (1 - 0.20*(1 - f))`
  * `tau_eff = tau_base * (1 + 0.50*(1 - f))`
* Interpretation: low fatigue (f≈1) ≈ base; more fatigue increases latencies and reduces speed/accel.

```python
GK_R = 3.0
GK_V_DIVE = 6.0
```

* GK reach radius and dive speed defaults, only used when GK flagged.

---

# column maps (schema adapters)

```python
COLS = {...}
PHYS_COLS = {...}
```

* Canonical names used inside the script mapped to expected FIFA column labels.
* If the feeds change labels, you adjust here; rest of logic remains stable.

---

# helpers

## `kmh_to_ms(v_kmh)`

* Converts km/h → m/s. Returns `np.nan` if input is NaN.

## `clamp(x, lo, hi)`

* Hard bounds a scalar value.

## `calc_t85(n_sprints)`

```python
n = 0 if NaN else float(n_sprints)
t85 = clamp(2.2 - 0.01 * n, 1.6, 2.2)
```

* A proxy for **time to 85% top speed**. More sprints ⇒ better acceleration ⇒ smaller `t85` (bounded to \[1.6, 2.2] s).

---

# deriving base physical parameters per player

## `derive_phys_params(phys_df)`

**Inputs:** A concatenation of all physical summary sheets (can be multi-match).

**Steps:**

1. **Team medians for fallback speed**:

   ```python
   for team, sub in phys_df.groupby("Team Name"):
       valid = sub["Max Speed (km/h)"].dropna().astype(float)
       v_by_team[team] = kmh_to_ms(median(valid)) if any valid else DEFAULT_V_MAX_MS
   ```

   * Used when a specific player lacks `Max Speed`.

2. **Per-player loop**:
   For each row `r`:

   * `pid`, `pname`, `team`, `n_sprints`, `v_kmh` pulled from `PHYS_COLS`.
   * `v_max`:

     * If `v_kmh` present: `kmh_to_ms(v_kmh)`.
     * Else: team median (m/s) from `v_by_team`, else `DEFAULT_V_MAX_MS`.
   * `t85 = calc_t85(n_sprints)`.
   * `a_max = min(0.85 * v_max / t85, 6.5)`.

     * 0.85 factor says “accel phase targets \~85% v\_max”.
   * `d_max = min(1.25 * a_max, 8.0)`.

     * Braking > acceleration by 25%, capped at 8.0 m/s².
   * Latencies (smaller with more sprints, bounded):

     * `n = 0 if NaN else n_sprints`
     * `tau_react = clamp(0.26 - 0.002*min(20, n), 0.18, 0.30)`
     * `tau_turn  = clamp(0.34 - 0.002*min(20, n), 0.26, 0.40)`
   * Store into `params[(player_id, player_name)] = {...}` with GK fields initialized to `False/None`.

**Output:** dict key = `(player_id or None, player_name or "")` → dict of base caps/taus.

---

# event tagging: regex and detectors

## Regular expressions

```python
CONTACT_RE = re.compile(r"(foul|tackle|aerial|duel|collision|challenge)", re.IGNORECASE)
SAVE_RE    = re.compile(r"(save|goal\s?kick|keeper|gk)", re.IGNORECASE)
```

* Loose, case-insensitive matching of body-contact / GK cues from free-form columns.

## `detect_contact(row) -> float`

* Builds `text` by concatenating `["event", "event_type", "body_type"]` if present.
* If `CONTACT_RE` does **not** match: returns `0.0`.
* If matches:

  * If contains `aerial|header`: **0.4**
  * If contains `foul|collision`: **0.7** (heavy contact)
  * Else (tackle/duel/challenge): **0.4**

**Rationale:** coarse intensity mapping; heavy contact triggers slowed recovery later.

## `carry_displacement_intensity(row) -> float`

* Only runs if `event_type == "carry"`.
* Reads start/end coords: `x_location_start/end`, `y_location_start/end`.
* **Coordinate scaling**:

  * If max(|x|) ≤ 1.2 → treat as **normalized**, scale to meters by `(105, 68)`.
  * Else treat as **meters** (scale 1.0).
* Compute displacement `d = hypot(dx, dy)` and bin:

  * `d ≥ 12 m` → **1.0**
  * `8 ≤ d < 12` → **0.6**
  * `4 ≤ d < 8` → **0.3**
  * else → **0.0**

**Meaning:** longer carries are stronger “bursts”, costing **more** fast fatigue.

## `choose_time_ms(row) -> Optional[int]`

* Returns the first available integer among:

  * `event_end_time_in_ms`
  * `match_run_time_in_ms`
  * `match_time_in_ms`
* If none parseable → `None`.

**Purpose:** unify timing across feeds for sorting & recovery evolution.

## `detect_gks_from_events(events) -> dict[(pid,name) -> bool]`

* Scans rows; if `SAVE_RE` hits in `event`/`event_type`, marks `(pid,name)` as GK.
* If **none** found:

  * Looks for `from_player_shirt_number` in `{1,13}` (typical GK shirt numbers). Marks the first occurrence.
* Returns a dict mapping `(pid,name)` → `True/False`.

**Note:** very lightweight; good enough for U17 demo; can be replaced by role metadata if available.

---

# the generator core

## `generate_synthetic_sidecar(events, phys_df_concat=None) -> DataFrame`

**Inputs:**

* `events`: a single match’s events DataFrame (first sheet of the events Excel).
* `phys_df_concat`: (optional) a concatenation of *all* physical summary rows found in `Phys_Summary/`.

**Precomputation:**

* `phys_params_map`:

  * If `phys_df_concat` exists: `derive_phys_params(...)`.
  * Else: empty dict; we’ll synthesize per player from team medians/defaults.
* `gk_map = detect_gks_from_events(events)`.
* `team_v_defaults`:

  * If any physical summaries are present: for each team, compute median `Max Speed (km/h)` → convert to m/s.
  * Else: empty dict (fall back to `DEFAULT_V_MAX_MS`).

**Per-player rolling state:**

```python
states = defaultdict(lambda: dict(
  F_fast=1.0, F_slow=1.0, last_time_ms=None, tau_mult_until_ms=0, recent_bursts=[]
))
```

* Auto-initializes upon first access for a `(pid,name)` key.

**Sorting events temporally:**

```python
events_sorted = events.copy()
events_sorted["__time_ms"] = events_sorted.apply(choose_time_ms, axis=1)
events_sorted = events_sorted.sort_values(["match_id", "__time_ms", "event_id"], kind="mergesort")
```

* Uses **stable mergesort** to preserve event order when times tie.
* Ensures recovery is integrated in the correct timeline.

**Loop over events (row by row):**
For each `row` in `events_sorted`:

1. **Keys:**

   ```python
   mid = row["match_id"]; eid = row["event_id"]
   pid = row["from_player_id"]; pname = row["from_player_name"]
   team = row["team_name"]
   key = (pid or None, pname or "")
   ```

   * This is the **actor** (from\_player) whose state we update and for whom we output a synthetic row.

2. **Base physical params (`pp`):**

   * Try `phys_params_map[key]` first.
   * If missing:

     * `v_max = team_v_defaults.get(team, DEFAULT_V_MAX_MS)`
     * Assume **8 sprints** for default personalization:

       * `t85 = clamp(2.2 - 0.08, 1.6, 2.2) = 2.12`
       * `a_max = min(0.85*v_max/t85, 6.5)`
       * `d_max = min(1.25*a_max, 8.0)`
       * `tau_react = clamp(0.26 - 0.002*8, 0.18, 0.30) = clamp(0.244, 0.18, 0.30)`
       * `tau_turn  = clamp(0.34 - 0.002*8, 0.26, 0.40) = clamp(0.324, 0.26, 0.40)`
     * Build `pp` dict with GK fields empty.

3. **GK extras (if applicable):**

   ```python
   is_gk = gk_map.get(key, False)
   if is_gk and not pp["is_gk"]:
       pp["is_gk"] = True
       pp["tau_gk_react"] = pp["tau_react"] + 0.02
       pp["r_gk"] = 3.0
       pp["v_dive"] = 6.0
   ```

   * One-time augmentation: adds GK reaction base and dive parameters.

4. **Time bookkeeping & recovery:**

   ```python
   t_ms = row["__time_ms"]
   st = states[key]                # player's rolling state
   if st["last_time_ms"] is None: st["last_time_ms"] = t_ms
   dt_s = max(0.0, (t_ms - st["last_time_ms"]) / 1000) if both present else 0.0
   ```

   * Compute time elapsed since this player’s **own** last event (not since last global event).

   **Recovery step (exponential towards 1.0):**

   ```python
   tau_mult = 1.0
   if st["tau_mult_until_ms"] and t_ms < st["tau_mult_until_ms"]:
       tau_mult = 1.5  # slowed recovery after heavy contact
   tau_fast = 45 * tau_mult
   tau_slow = 600 * tau_mult

   st["F_fast"] = 1 - (1 - st["F_fast"]) * exp(-dt_s / tau_fast)
   st["F_slow"] = 1 - (1 - st["F_slow"]) * exp(-dt_s / tau_slow)
   ```

   * If no time passed (`dt_s=0`), values unchanged.

5. **Burst and pressure:**

   ```python
   I_burst = carry_displacement_intensity(row)  # 0/0.3/0.6/1.0 from carry length
   pres = str(row.get("pressure","")).lower().strip() in {"1","true","yes","y","t"}
   if pres:
       I_burst = max(I_burst, 0.4)  # pressure ensures at least moderate acute load
   ```

6. **Quick chain (same player acts again within 5s):**

   ```python
   last_event_time = defaultdict(lambda: None)  # outside loop
   last_t = last_event_time.get(key)
   quick_chain = 0
   if last_t is not None and t_ms is not None and (t_ms - last_t) <= 5000:
       I_burst += 0.3
       quick_chain = 1
   ```

   * Captures back-to-back involvements as extra burst cost.

7. **30s rolling accumulation (I\_burst\_long):**

   ```python
   st["recent_bursts"].append((t_ms, I_burst))
   # keep only entries within 30,000 ms of current time
   st["recent_bursts"] = [(tm, ib) for (tm, ib) in st["recent_bursts"] if t_ms - tm <= 30000]
   sum_bursts = sum(ib for _, ib in st["recent_bursts"])
   I_burst_long = clamp(sum_bursts / 30.0, 0.0, 1.0)
   ```

   * Heuristic normalization: if you did “1.0 of burst” every second, 30s would sum to 30 → normalize to 1.0.

8. **Contact intensity:**

   ```python
   C = detect_contact(row)  # 0.0 (none), 0.4 (aerial/duel), 0.7 (foul/collision)
   ```

9. **Event-time decrements (apply *after* recovery):**

   ```python
   st["F_fast"] = clip( st["F_fast"] - (0.20*I_burst + 0.30*C), 0, 1 )
   st["F_slow"] = clip( st["F_slow"] - (0.10*I_burst_long + 0.20*C), 0, 1 )
   ```

   * Acute burst hits **fast** more; accumulated & contact affect **slow** as well.

   **Slow-recovery window after heavy contact:**

   ```python
   if C >= 0.7 and t_ms is not None:
       st["tau_mult_until_ms"] = t_ms + 60000
   ```

10. **Final fatigue multiplier and effective kinematics:**

    ```python
    f_i = 0.6*F_fast + 0.4*F_slow

    v_i_eff        = v_max    * (1 - 0.25*(1 - f_i))
    a_i_eff        = a_max    * (1 - 0.40*(1 - f_i))
    d_i_eff        = d_max    * (1 - 0.20*(1 - f_i))
    tau_react_eff  = tau_react* (1 + 0.50*(1 - f_i))
    tau_turn_eff   = tau_turn * (1 + 0.50*(1 - f_i))

    # GK tweak on reaction base:
    if pp["is_gk"] and pp.get("tau_gk_react") is not None:
        tau_react_eff = pp["tau_gk_react"] * (1 + 0.50*(1 - f_i))
    ```

11. **Emit one output row per event:**
    Contains keys, instantaneous reservoirs, burst/contact, base caps, GK extras, and effective values.

12. **Update timestamps:**

    ```python
    if t_ms is not None:
        last_event_time[key] = t_ms
        st["last_time_ms"]   = t_ms
    ```

**After the loop:**

* Build a DataFrame `sidecar` from `out_rows`.
* Coerce numeric columns (those starting with `syn_`, ending with `_eff`, `_max`, or starting with `tau_`) to `float`.
* Return the sidecar.

---

# physicals loading & batch runner

## `load_all_phys(phys_dir)`

* Scans the `Phys_Summary/` folder for all `.xlsx/.xls`.
* Reads the **first sheet** of each; concatenates vertically (ignore index).
* Returns `None` if none found (the generator then uses team defaults/fallbacks).

## `run_batch(base_dir=".", events_subdir="Events", phys_subdir="Phys_Summary", synthetic_subdir="Synthetic")`

* Builds absolute paths to the three folders from `base_dir`.
* `os.makedirs(out_dir, exist_ok=True)` ensures `Synthetic/` exists.
* `phys_df_concat = load_all_phys(phys_dir)` once for all matches.
* Find all events spreadsheets under `Events/`.
* For each events file:

  * `ev = pd.read_excel(ev_path, sheet_name=0)`
  * `sidecar = generate_synthetic_sidecar(ev, phys_df_concat)`
  * Write to `Synthetic/<events_basename>_synthetic.csv`.

**The `__main__` block** (in your paste):

```python
if __name__ == "__main__":
    run_batch(base_dir="/mnt/data")
```

* On your machine, set this to your repo root:

  ```python
  run_batch(base_dir="/Users/jessicafan/LIM")
  ```

  or add a CLI arg parser if you prefer.

---

# outputs: exact fields & provenance

For **each event row where a `from_player_*` exists**, the sidecar includes:

**Keys**

* `match_id`, `event_id`, `from_player_id`, `from_player_name`, `team_name`
  (copied from events row to preserve joinability)

**Instantaneous fatigue**

* `syn_F_fast`, `syn_F_slow` — post-recovery, post-decrement reservoirs ∈ \[0,1]
* `syn_f_i` — blended fatigue multiplier = `0.6*F_fast + 0.4*F_slow`

**Event stressors**

* `syn_I_burst` — acute burst intensity in this event (carry length bins; at least 0.4 if under pressure; +0.3 if quick chain)
* `syn_I_burst_long` — normalized 30s rolling sum of `I_burst` (∈\[0,1])
* `syn_contact` — contact intensity from regex mapping (0.0/0.4/0.7)
* `syn_quick_chain` — 1 if same player acted within 5s, else 0

**Base physical caps/latencies (pre-fatigue)**

* `v_i_max` — from Max Speed (km/h)/3.6 if available; else team median; else 8.6 m/s
* `a_i_max` — `min(0.85 * v_i_max / t85, 6.5)` where `t85 = clamp(2.2 - 0.01 * #Sprints, 1.6, 2.2)`
* `d_i_max` — `min(1.25 * a_i_max, 8.0)`
* `tau_i_react` — `clamp(0.26 - 0.002*min(20, #Sprints), 0.18, 0.30)`
* `tau_i_turn`  — `clamp(0.34 - 0.002*min(20, #Sprints), 0.26, 0.40)`
* `is_gk` — {0,1} from event text/jersey heuristic
* `tau_GK_react` — `tau_i_react + 0.02` if GK else NaN
* `r_GK` — 3.0 if GK else NaN
* `v_dive` — 6.0 if GK else NaN

**Effective (post-fatigue)**

* `v_i_eff`        — `v_i_max * (1 - 0.25*(1 - syn_f_i))`
* `a_i_eff`        — `a_i_max * (1 - 0.40*(1 - syn_f_i))`
* `d_i_eff`        — `d_i_max * (1 - 0.20*(1 - syn_f_i))`
* `tau_i_react_eff`— if GK: `tau_GK_react * (1 + 0.50*(1 - syn_f_i))`, else `tau_i_react * (...)`
* `tau_i_turn_eff` — `tau_i_turn * (1 + 0.50*(1 - syn_f_i))`

---

# assumptions & edge cases

* **Coordinates**: If any of the x (or y) inputs in a carry have absolute values ≤ 1.2, treat as normalized and scale by (105, 68). Otherwise, assume meters. This allows mixing normalized and metric feeds.
* **Timing**: If `event_end_time_in_ms` missing, we fall back to `match_run_time_in_ms`, then `match_time_in_ms`. If all missing, event contributes no recovery but still emits a row with current state.
* **Pressure**: Accepts multiple encodings (bool/0/1/"true"/"False"/etc.).
* **Players missing in physicals**: fall back to team median speed or global default; sprints proxy = 8 to personalize default caps/taus.
* **GK detection**: heuristic; prefer to swap to explicit role fields if your feed has them.
* **Multiple matches in phys\_df\_concat**: That’s fine — the code uses **player identity** (ID+name) and **team medians**; not per-match slicing.
* **Sorting**: stable mergesort by (`match_id`, `__time_ms`, `event_id`) ensures deterministic order for same-timestamp events.

---

# why these design choices

* **Two-reservoir fatigue**: Captures short-term and long-term load with simple, explainable exponentials.
* **Event-only burst/contact**: Uses only what’s guaranteed in event feeds (no tracking). Displacement carries, pressure flags, and contact keywords approximate load reasonably for a coach demo.
* **Personalization from physicals**: Speed & sprint counts are simple, available signals to individualize caps and latencies.
* **GK extras**: Slightly higher reaction baseline and dive traits produce sensible behavior in pitch control without full shot-stopping physics.
* **Effective scaling** (`K_*`): As fatigue increases (f↓), speed/accel drop and latencies rise with conservative slopes to avoid extreme swings.

---

# modding cheat-sheet (where to tweak)

* **Fatigue sensitivity**: `ALPHA_*`, `BETA_*` (bigger → drains faster)
* **Recovery speeds**: `TAU_FAST_BASE`, `TAU_SLOW_BASE`
* **Heavy contact effect**: `HEAVY_CONTACT_THRESHOLD`, `HEAVY_CONTACT_TAU_MULT`, `HEAVY_CONTACT_WINDOW_S`
* **Carry burst bins**: thresholds (4/8/12 m) and intensities (0.3/0.6/1.0) in `carry_displacement_intensity`
* **Pressure minimum burst**: `I_burst = max(I_burst, 0.4)`
* **Quick chain window/boost**: 5s & +0.3
* **Effective scaling slopes**: `K_V`, `K_A`, `K_D`, `K_TAU`
* **Caps & latencies from physicals**: formulas in `derive_phys_params` (change 0.85 factor, caps 6.5/8.0, latency slopes, etc.)
