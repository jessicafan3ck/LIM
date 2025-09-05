# 0) Imports & Types

* `math`, `json`: math utilities and (optionally) JSON formatting.
* `dataclasses.dataclass`, `field`: declarative data containers for configs/state.
* `typing`: for explicit type hints (clarifies intent; no runtime effect).
* `collections.defaultdict`, `Counter`: counting and default-valued dicts.
* `numpy as np`, `pandas as pd`: vectorized numerics and tabular IO.

---

# 1) Utility / Math helpers

### `logistic(x) -> float`

* **Input**: scalar or array-like `x` (dimensionless).
* **Output**: `1/(1+e^{−x})` in (0,1).
* **Use**: Convert linear score (logit) to probability in success models.

### `clamp(x, lo, hi) -> float`

* **Input**: `x` value, lower/upper bounds `lo`, `hi`.
* **Output**: `x` clipped into `[lo, hi]`.
* **Use**: Keep indices/probabilities/coordinates within safe ranges.

### `unit(v: np.ndarray) -> np.ndarray`

* **Input**: 2D vector `v`.
* **Output**: unit vector `v/||v||` or `[0,0]` if zero norm.
* **Use**: (Currently unused in your pasted code but standard helper for direction.)

### `softmax(x, temp=1.0) -> np.ndarray`

* **Input**: scores `x`, temperature `temp>0`.
* **Process**:

  * Convert to `float` array.
  * Divide by `temp` (higher temp → softer distribution).
  * Subtract max for numerical stability.
  * Exponentiate & normalize.
* **Output**: probability vector same shape as `x`.
* **Use**: (Not used in final code path; policy selection uses precomputed categorical `p`.)

### `entropy(p) -> float`

* **Input**: probability vector `p` (elements ≥ 0, sum \~ 1).
* **Process**:

  * Filter out zeros.
  * Compute `−∑ p log p` (natural log).
* **Output**: scalar entropy (0 for delta, up to `log K` for uniform over `K`).
* **Use**: Gauge “peakiness” of policy distributions → proxy for data support.

### `geometric_mean(vals, eps=1e-9) -> float`

* **Input**: positive values (probabilities/confidences).
* **Process**:

  * Clip to `[eps,1]` to avoid log(0).
  * Return `exp(mean(log(vals)))`.
* **Use**: Combine model/state/data confidences multiplicatively in a numerically stable way.

---

# 2) Data schema constants

### `REQ_COLS`

* **Purpose**: Hard gate. `fit_match_model` will throw if any missing.
* **Columns**:

  * Identifiers: `match_id`, `event_id`.
  * Actor/team: `team_name`, `from_player_id`, `from_player_name`.
  * Geometry (start/end): `x_location_start`, `y_location_start`, `x_location_end`, `y_location_end`.
  * Semantics: `event_type`, `event`, `pressure`, `half_time`.
  * Synthetic/effective physiology: `syn_f_i`, `v_i_eff`, `a_i_eff`, `d_i_eff`, `tau_i_react_eff`, `tau_i_turn_eff`.
* **Logic**: The simulator expects an *already-merged* events+synthetic dataframe with these names.

### `OPT_COLS`

* **Optional** hints for richer modeling:

  * Line-break metadata, body/technique tags, and times: `event_end_time_in_ms`, `match_run_time_in_ms`.
* **Use**: `fit_ball_model` uses times; success fit uses body/style if available.

---

# 3) Dataclasses (configuration & state)

### `FieldConfig`

* **Fields**:

  * `length_m = 105.0`, `width_m = 68.0` (FIFA pitch default).
  * Grid resolution: `nx = 36`, `ny = 24` (cells).
* **`cell_size()`**: returns `(dx, dy)` in meters: `(105/36, 68/24)`.
* **Use**: discretization for control/hazard fields.

### `BallModel`

* **Fields**:

  * `kappa_ground = 1.3 s⁻¹`: crude exponential decay parameter for ground balls.
  * `ctrl_delay_base = 0.18 s`: first-touch/control latency.
  * `ctrl_delay_pressure_mult = 1.35`: multiplicative penalty under pressure.
* **`control_delay(under_pressure)`**:

  * Returns `ctrl_delay_base` if false, else `ctrl_delay_base*1.35`.
* **Use**: Advance time between events.

### `PolicyModel`

* **Fields**:

  * `actions = ["pass","carry","takeon","shot","clear"]`: action vocabulary.
  * `zone_phase_probs`: dict keyed by `(ix, iy, phase)` each mapping to `np.ndarray p` same length as `actions`.
  * `temperature`: not applied in current sampling (but could modulate entropy later).
* **Use**: Sample action types conditioned on location & phase.

### `SuccessModel`

* **Fields**:

  * `params`: action → coefficient dicts, e.g. `{ "pass": {"b0":..., "b_d":..., ...}, ... }`.
  * `p_min=0.05`, `p_max=0.98`: clamp probabilities—avoid degenerate 0/1.
* **Use**: Turn simple features (distance, pressure, aerial) into success probabilities via logistic regression–style scores.

### `MatchFit`

* **Fields** (fitted per match):

  * `field: FieldConfig`
  * `ball: BallModel`
  * `policy: PolicyModel`
  * `success: SuccessModel`
  * `grid_centers: np.ndarray` of shape `(nx, ny, 2)` with each cell center `(x,y)` in meters.

### `PlayerSnapshot`

* **A single player’s state at the seed**:

  * Identity: `player_id: Optional[int]`, `player_name: str`, `team_name: str`.
  * Synthetic physiology/effects: `f_i`, `v_eff`, `a_eff`, `d_eff`, `tau_react_eff`, `tau_turn_eff`.
    **All are floats** (units: `v_eff` m/s, `a_eff` `m/s²`, `taus` s).
  * Position: `x`, `y` (meters on 105×68 pitch).

### `GameState`

* **Fields**:

  * `t_ms: int` current simulation time in milliseconds.
  * `half_time: int` (1 or 2 typically).
  * `ball_xy: (float, float)` ball position in meters.
  * `possession_team: Optional[str]` current team in possession (string name).
  * `carrier_id: Optional[int]` player ID currently with the ball (if known).
  * `players: Dict[(player_id, player_name) -> PlayerSnapshot]` roster snapshot.
  * `last_event_row: Optional[pd.Series]` seed event (for “phase”).
  * Caches: `control_field: Optional[np.ndarray (nx,ny)]`, `hazard_field: Optional[np.ndarray (nx,ny)]`.

---

# 4) Fitting functions

### `_grid_centers(field) -> np.ndarray`

* **Compute**:

  * `dx = 105/nx`, `dy = 68/ny`.
  * `xs = (arange(nx) + 0.5) * dx` → cell-center x for each i∈\[0..nx-1].
  * `ys = (arange(ny) + 0.5) * dy` → center y for j∈\[0..ny-1].
  * `X, Y = np.meshgrid(xs, ys, indexing="ij")`: shapes `(nx, ny)` each.
  * `g = stack([X, Y], axis=-1)`: shape `(nx, ny, 2)`, units meters.
* **Use**: lookups when computing distance to cells for control/hazard.

### `fit_ball_model(events) -> BallModel`

* **Goal**: try to infer `kappa_ground` from passes with timing; else default to 1.3.
* **Steps**:

  1. Copy events → `df`.
  2. Filter `is_pass = (event_type.lower()=="pass")`.
  3. Keep rows with all four coords present.
  4. If both `match_run_time_in_ms` and `event_end_time_in_ms` exist:

     * Convert to numeric, filter `t1 > t0`.
     * Compute `dx, dy` in dataframe (float).
     * If their magnitudes look normalized (≤1.2), scale to meters: `*105`/`*68`.
     * Distance `d = hypot(dx, dy)`.
     * Duration `dt = (t1−t0)/1000` (seconds).
     * Launch speed proxy `v0 = d/dt`; keep only `(0,40)` m/s to exclude junk.
     * If enough samples (≥10–20), **currently** just returns `kappa=1.3`.
       (The comment notes a future regression; for now the mid default is used.)
* **Outputs**: `BallModel(kappa_ground=1.3)` plus default control delays.

### `_zone_index(field, x, y) -> (ix, iy)`

* **Input**: a point in meters.
* **Safety**: if `x` or `y` not finite → return center cell `(nx//2, ny//2)`.
* **Computation**:

  * `ix = floor(x / (length/nx))` clamped `[0, nx-1]`.
  * `iy = floor(y / (width/ny))` clamped `[0, ny-1]`.
* **Use**: zone/phase keys for policy lookups and control/hazard queries.

### `fit_policy(events, field) -> PolicyModel`

* **Goal**: empirical action distribution by grid cell & phase.
* **Steps**:

  1. `pm = PolicyModel()` with canonical action set.
  2. `phase = lambda r: str(r.get("sequence_type","open")).lower()`.
  3. Iterate rows:

     * Try to read `x_location_start`, `y_location_start` → skip if invalid.
     * `ix,iy = _zone_index(...)`.
     * Read action `a = event_type.lower()`, normalize synonyms:

       * `"dribble"` → `"carry"`, `"take-on"` → `"takeon"`, `"shoot"` → `"shot"`, `"clearance"` → `"clear"`.
       * Skip if not in the canonical vocab.
     * Increment `counts[(ix,iy,phase)][a] += 1`.
  4. Convert each counter to a probability vector over `pm.actions`:

     * `p[a] = count/total` with a tiny `+1e-6` smoothing to avoid exact zeros; renormalize.
     * Store under `pm.zone_phase_probs[(ix,iy,phase)] = p`.
* **Output**: learned categorical priors for sampling actions.

### `_bin_stats(vals, bins) -> (centers, means)`

* **Utility**: For binned calibration heuristics.
* **Steps**:

  * Digitize `vals` over `bins`; compute mean per bin.
  * Return bin centers and bin means (NaN where empty).

### `fit_success_models(events) -> SuccessModel`

* **Goal**: Lightweight, within-match, monotone-ish heuristics for success probability per action type.
* **Preprocessing**:

  * `ev = events.copy()`.
  * `pressure_bin = 1` if `pressure` in {1,true,t,yes,y}, else 0.
  * `aerial_bin = 1` if any of `body_type` contains “Header” or `style/style_additional` contains High/Cross/Chip/Long Ball.
  * Compute distances:

    * `dx = x_end − x_start`, `dy = y_end − y_start`.
    * If deltas look normalized, scale to meters.
    * `dist_m = hypot(dx, dy)`.
  * Outcomes:

    * If no explicit `outcome`, then define for pass/carry/takeon as `1` if end coords exist, else `0`; shots default later.
* **Pass model**:

  * Subset `event_type=="pass"` and `outcome ∈ {0,1}`.
  * If ≥30 samples:

    * Bin by distance; estimate near vs far completion rates to set distance slope `b_d` (near−far).
    * `p_base = mean(outcome)` → baseline logit `b0 = log(p/(1-p))`.
    * Penalties: `b_press = -0.3`, `b_aer = -0.1` (hand-tuned defaults).
    * Store `params["pass"] = {b0, b_d, b_press, b_aer}`.
* **Carry model**:

  * `event_type=="carry"` with outcomes.
  * If ≥20 samples:

    * Similar binned slope `b_d`.
    * `b0` from mean outcome; `b_press = -0.25`.
* **Take-on model**:

  * `event_type ∈ {"takeon","take-on"}` with outcomes.
  * If ≥20 samples:

    * Baseline `p_base` (fallback 0.45 if no outcomes), `b_press=-0.3`.
* **Shot model**:

  * `event_type=="shot"`.
  * If ≥10 samples:

    * If no outcome present, set all to 0 (placeholder for goals).
    * Binning only used to set plausible `p_base` guard; main slope fixed `b_dist=-0.08` (distance penalty).
    * `b0` from guarded `p_base` (min 0.02, max 0.2).
* **Output**: `SuccessModel` with possibly some actions missing (if too sparse).

---

# 5) Control & Hazard fields

### `control_field(field, players) -> np.ndarray (nx,ny)`

* **Goal**: Fast proxy for “who controls where” based on time-to-reach (TTR).
* **Steps**:

  1. `g = _grid_centers(field)` → `(nx,ny,2)` cell centers.
  2. `teams = sorted({p.team_name for players})` (unique names; expect 2).
  3. For each team, init `ttr[team] = big matrix (nx,ny) filled 1e6`.
  4. For each player `p`:

     * `v = max(0.1, p.v_eff)` (m/s), `tau = max(0.01, p.tau_react_eff)` (s).
     * `dist = hypot(g[:,:,0]−p.x, g[:,:,1]−p.y)` (meters).
     * `ttr_p = tau + dist / v` (seconds).
     * `ttr[team] = minimum(ttr[team], ttr_p)` (best responder per cell per team).
  5. If not exactly 2 teams: return zeros (no advantage defined).
  6. `adv = ttr[A] − ttr[B]`:

     * Positive ⇒ Team A reaches faster (smaller time) → actually **note**: “A faster” would be `ttr[B] − ttr[A] < 0`. Here we define positive as A minus B; then we squash:
     * `adv = tanh(adv/0.8)`, so roughly in `[-1,1]`.
* **Interpretation**: Sign encodes which team is faster to arrive; magnitude is sharpness.

### `hazard_field(field, players, pressure_prior=0.25) -> np.ndarray (nx,ny)`

* **Goal**: Dense-crowding proxy for pressure (bigger near players).
* **Steps**:

  1. `g = _grid_centers(field)`.
  2. Gather all player `(x,y)` → `positions`.
  3. Init `dmin` (nx,ny) to a large number (50 m).
  4. For each player, compute `dist = hypot(cell−player)` and `dmin = minimum(dmin, dist)`.
  5. Hazard = `1/(1 + dmin)` in (0,1], then add prior 0.25 and clip to `[0,1]`.
* **Interpretation**: Near players ⇒ higher hazard; empty space ⇒ lower hazard with floor `~0.25`.

---

# 6) Success scoring

### `_success_prob(action, features, sm) -> float in [0.05,0.98]`

* **Inputs**:

  * `action`: one of `pass/carry/takeon/shot`.
  * `features`:

    * `dist_m`: meters to target (continuous).
    * `pressure`: 0/1 flag (from hazard threshold).
    * `aerial`: 0/1 pass height proxy (random 15% if pass).
  * `sm`: `SuccessModel` (may or may not have params for this action).
* **Process**:

  1. If `action` not in `sm.params`, return default 0.5 (then clamped).
  2. Score `z` starts at `b0`.
  3. Add penalties/bonuses:

     * `pass`: `z += b_d*(-dist) + b_press*(-pressure) + b_aer*(aerial)`.
     * `carry`: `z += b_d*(-dist) + b_press*(-pressure)`.
     * `takeon`: `z += b_press*(-pressure)`.
     * `shot`: `z += b_dist*(dist)`. (Note sign: `b_dist` is negative.)
  4. Convert `z` with `logistic(z)`, then clamp to `[p_min, p_max]`.
* **Output**: scalar probability used to draw success/failure.

---

# 7) Policy sampling

### `_phase(row) -> str`

* **Input**: last event row or `None`.
* **Output**: phase string, default `"open"` (open play).
* **Use**: Key for `(ix,iy,phase)` policy lookup.

### `sample_action_probs(ix, iy, phase, policy) -> (actions, p)`

* **Lookup**: `(ix,iy,phase)` in `policy.zone_phase_probs`.
* **If found**: return stored `p` over canonical `actions`.
* **Else**: uniform `1/|actions|`.

### `sample_action(...) -> str`

* **Use**: RNG draw from the above categorical.

---

# 8) Simulator

### `SimConfig`

* **Fields**:

  * `horizon_K`: number of steps to simulate (default 5).
  * `rollouts_R`: Monte Carlo paths (default 200).
  * `half_life_events`: horizon-decay for sequence confidence (events where confidence halves).
  * `topM`: aggregation cap (used when summarizing patterns).
  * `seed`: RNG seed for reproducibility.

### `_confidence_step(p_succ, state_sharp, data_support) -> float`

* **Inputs**:

  * `p_succ` (model certainty about this action outcome),
  * `state_sharp` = |control advantage| at (ix,iy) ∈ \[0,1].
  * `data_support`: 1 − normalized entropy of policy probs at this (ix,iy,phase).
* **Process**:

  * `c_model = max(p_succ, 1 − p_succ)` (peakedness).
  * `c_state = clamp(|adv|,0,1)`.
  * `c_data = clamp(data_support,0,1)`.
  * Geometric mean: balances the three (any near 0 drags it down).
* **Output**: per-step confidence multiplier (0–1).

### `_state_sharpness(ctrl_adv_here) -> float`

* **Simply** `abs(ctrl_adv_here)`.

### `_data_support(action, ix, iy, phase, policy) -> float`

* **If** policy has a distribution at `(ix,iy,phase)`:

  * Compute entropy `H`, maximum entropy `Hmax=log(K)`, return `1 − H/Hmax` ∈ \[0,1].
  * Intuition: more peaked (low entropy) ⇒ higher support.
* **Else** return `0.2` fallback.

### `_action_geometry(action, carrier, field, rng) -> dict`

* **Goal**: sample a plausible target `(tx,ty)` and compute `dist_m`.
* **Cases**:

  * `"pass"`:

    * Radius `r∈U[6,22]` meters, angle `θ∼N(0, π/6)` (forward bias).
    * `(tx,ty)` = carrier + (r·cosθ, r·sinθ), clamped to pitch.
    * `aerial` set to `0.0` here; separate `aerial` feature is sampled later with 15% chance.
  * `"carry"`:

    * `r∈U[3,12]`, `θ∼N(0, π/8)` (shorter and narrower than pass).
  * `"takeon"`:

    * `r∈U[2,6]`, `θ∼N(0, π/5)` (small advance).
  * `"shot"`:

    * Target goal center at `(length, width/2)`; distance computed to goal.
  * `"clear"`:

    * Keep `x` same, bump `y` up or down by `U[8,20]` meters, clamped to `[0,width]`.
* **Output**: dict with keys `tx, ty, dist_m` (and `aerial` for pass path).

### `_ixiy(field, x, y)`

* Same as `_zone_index` but returns center cell on bad coords right away.

### `simulate_rollouts(initial_state, fit, cfg) -> dict`

**High-level**: Monte Carlo over future event sequences (length `K`), using fitted policy to choose actions, `SuccessModel` to draw outcomes, and simple ball/player updates to evolve state. Confidence is aggregated multiplicatively with horizon decay.

**Detailed flow**:

1. **Setup**:

   * `rng = default_rng(cfg.seed)`.
   * `field = fit.field`.
   * `lambda_ = ln(2)/half_life_events` (so confidence decays by ½ every `half_life_events`).
   * Ensure fields cached:

     * If `initial_state.control_field is None`, compute via `control_field`.
     * If `initial_state.hazard_field is None`, compute via `hazard_field`.

2. **Rollouts loop** (`r = 1..R`):

   * `s = initial_state`. (Note: shallow reuse; the code mutates `s`. For strict independence across rollouts you’d deep copy, but given simple use it’s okay for MVP if each rollout re-derives from same initial references and mutations are local to the loop body. If you plan to parallelize or retain per-rollout state post-hoc, consider copying.)

   * Initialize `seq=[]`, `confs=[]`, `t_ms = s.t_ms`.

   * **Find current carrier**:

     * If `carrier_id` not present or key not found:

       * Pick nearest player to current `ball_xy` as carrier.
     * Else use `(carrier_id, name)` key.

   * **Steps loop** (`k = 1..K`):

     1. Read `carrier = s.players[carrier_key]`.
     2. Cell indices `ix, iy = _ixiy(field, carrier.x, carrier.y)`.
     3. `phase = _phase(s.last_event_row)` (typically `"open"`).
     4. **Sample action**: `action = sample_action(ix, iy, phase, fit.policy, rng)`.
     5. **Sample geometry**: `geom = _action_geometry(action, carrier, field, rng)` → `tx, ty, dist_m`.
     6. **Local pressure proxy**:

        * `ctrl_adv = s.control_field[ix,iy]` (∈\[−1,1]).
        * `hazard_here = s.hazard_field[ix,iy]` (∈\[0,1]).
        * `pressure = 1.0 if hazard_here > 0.5 else 0.0`.
     7. **Success probability**:

        * `feats = {"dist_m": dist_m, "pressure": pressure, "aerial": 1 with 15% chance if pass else 0}`.
        * `p_succ = _success_prob(action, feats, fit.success)`.
     8. **Bernoulli draw**: `success = (rng.random() < p_succ)`.
     9. **Per-step confidence**:

        * `c_model = max(p_succ, 1-p_succ)`.
        * `c_state = |ctrl_adv|`.
        * `c_data = 1 − H/Hmax` (if policy stats exist; else 0.2).
        * `c_step = geometric_mean([c_model, c_state, c_data])`.
        * Append to `confs`.
     10. **Advance state**:

         * Extract `tx, ty`.
         * If **success**:

           * `"pass"`:

             * Move ball to `(tx,ty)`.
             * New carrier = **nearest teammate** to `(tx,ty)` (search among `s.players` with same `team_name`).
           * `"carry"|"takeon"|"clear"`:

             * Move carrier to `(tx,ty)`; ball follows carrier.
           * `"shot"`:

             * Move ball to `(tx,ty)` (goal center); possession stays with shooter’s team (no outcome/goal logic yet).
           * Set `s.possession_team = carrier.team_name`.
         * Else (**failure**):

           * With probability 0.7 if action in {pass, takeon, carry}: **turnover**:

             * Flip possession to the “other” team (first team name not equal to carrier’s).
             * Ball to `(tx,ty)`.
             * New carrier = **nearest opponent** to `(tx,ty)`.
           * Otherwise (no turnover): jitter carrier position by ±1 meter approx (`2*(rand-0.5)`), clamp to pitch; ball at carrier.
     11. **Advance time**:

         * Crude travel speed proxy:

           * `travel_v = max(12, 18 * exp(-kappa*0.2))` (m/s), so around \~12–18 m/s.
         * `dt_travel = dist_m / travel_v`.
         * `dt_ctrl = fit.ball.control_delay(pressure)`.
         * `t_ms += int(1000*(dt_travel + dt_ctrl))`.
     12. **Refresh fields**:

         * Recompute `s.control_field = control_field(field, s.players)`.
         * Recompute `s.hazard_field = hazard_field(field, s.players)`.
     13. **Record step**:

         * Append dict with:

           * `k`, `actor_id`, `actor_name`, `team`, `action`, `success` (0/1), `p_succ`,
           * `tx`, `ty`, `dist_m`, `t_ms` (int),
           * `ctrl_adv`, `hazard_here`, `c_step`.

   * **Sequence confidence** (after K steps):

     * `C_seq = ∏_k c_step_k * exp(−lambda_ * K)`.

       * The `exp(−lambda_*K)` term enforces the half-life decay independent of per-step confidences.
     * Store `{"seq": seq, "C_seq": C_seq}` into `results`.

3. **Aggregate across rollouts**:

   * Count action patterns: `pat = tuple(a_k for k in 1..K)`; `Counter.most_common()` → `top_patterns`.
   * Compute `mean_sequence_confidence = mean(C_seq over rollouts)`.
   * `shot_prob_by_k[k] = (# of sequences where step k action == "shot") / R`.
   * Package **output dict**:

     * `"mean_sequence_confidence"`,
     * `"top_patterns"` (list of `[pattern_list, count]`, top 10),
     * `"shot_prob_by_k"` (dict of `k→prob`),
     * `"results"` (up to 50 sequences for readability).

**Notes / Design choices**:

* **Randomness**: Deterministic given `seed`. Change `seed` for varied samples.
* **Not training** end-to-end; `fit_success_models` is heuristic, local to match.
* **No hard physics**: Ball flight/control is simplified (it’s an MVP).
* **Turnover**: Only on failed pass/carry/takeon with 70% chance; shots don’t flip by themselves.
* **Possession**: Tracked at team level; carrier set by nearest search.

---

# 9) Fitting entry point

### `fit_match_model(merged_events) -> MatchFit`

* **Validation**: Ensure `REQ_COLS` present; else raise `ValueError`.
* **Normalize coordinates**:

  * Compute `max_abs` over all start/end x/y.
  * If `≤ 1.2`, treat as normalized \[0,1] → multiply x by 105, y by 68.
* **Fit**:

  * `field = FieldConfig()`.
  * `grid = _grid_centers(field)`.
  * `ball = fit_ball_model(df)` (uses time if available).
  * `policy = fit_policy(df, field)` (frequency model).
  * `success = fit_success_models(df)` (bin-heuristics).
* **Return**: `MatchFit(field, ball, policy, success, grid)`.

---

# 10) State construction

### `construct_state_from_row(row, team_rosters) -> GameState`

* **Inputs**:

  * `row`: a single merged event row at which you want to seed a simulation.
  * `team_rosters`: dict `team_name → list of (player_id, player_name, x, y)`.
    (You typically build this with a helper that infers \~11v11 positions from nearby events and pads with synthetic players if needed.)
* **Player snapshots**:

  * For every `(team, roster)` pair:

    * Create `PlayerSnapshot` with:

      * Identity & team from tuple.
      * Physiology/effects read from `row` (fallback defaults if missing):

        * `syn_f_i` (unitless 0–1 fatigue), `v_i_eff` (m/s), `a_i_eff`/`d_i_eff` (m/s²), `tau_*` (s).
      * Position `(x,y)` (meters) from roster tuple.
* **Ball & time**:

  * `bx, by` from `row["x_location_start"/"y_location_start"]` (numeric). If NaN: default to center `(52.5, 34.0)`.
  * `t_ms` from `match_run_time_in_ms` if present, else `event_end_time_in_ms`, cast to `int`, default `0`.
  * `half_time`: numeric cast, default `1`.
  * `carrier_id`: numeric cast of `from_player_id`, default `None`.
* **GameState**:

  * Fill `possession_team = row["team_name"]` (or `None`).
  * `players` is the dict you just built.
  * `last_event_row = row`.
  * `control_field`/`hazard_field` left `None` so the simulator computes them lazily at start.

---

# 11) Variable inventory (source • units • typical range)

* `FieldConfig.length_m/width_m` • config • meters • 105 / 68.
* `FieldConfig.nx/ny` • config • cells • 36 / 24.
* `_grid_centers` → `(nx,ny,2)` cell centers • meters • \~0–105 / 0–68.
* `BallModel.kappa_ground` • fitted (default) • s⁻¹ • \~1.3.
* `BallModel.ctrl_delay_base` • config • s • 0.18.
* `ctrl_delay_pressure_mult` • config • × • 1.35.
* `PolicyModel.zone_phase_probs[(ix,iy,phase)]` • fitted • probability vector over 5 actions • ∑=1.
* `SuccessModel.params[action]`:

  * Pass: `b0` (logit), `b_d` (per meter), `b_press` (~~−0.3), `b_aer` (~~−0.1).
  * Carry: `b0`, `b_d`, `b_press`.
  * Takeon: `b0`, `b_press`.
  * Shot: `b0`, `b_dist` (\~−0.08 per meter).
* `PlayerSnapshot.v_eff` • from synthetic sidecar • m/s • \~5–9.
* `PlayerSnapshot.tau_react_eff` • synthetic • s • \~0.2–0.4+.
* `control_field` • computed • (nx,ny) • `tanh((ttrA−ttrB)/0.8)` in \[−1,1].
* `hazard_field` • computed • (nx,ny) • `[0,1]` approx, \~0.25 baseline + crowding.
* `_action_geometry.dist_m` • sampled • meters • carry 2–12, pass 6–22, etc.
* `p_succ` per step • computed • \[0.05,0.98] after clamp.
* `c_step` per step • computed • \[0,1].
* `C_seq` per rollout • computed • small (product of c\_steps) × decay `exp(−λK)`.

---

# 12) Design rationale, failure modes, and tweak points

**Rationale**

* Keep the MVP interpretable: success models are simple logits with monotonic distance/pressure effects; policy is empirical frequencies; control/hazard are fast geometric proxies.
* Decouple modules so you can later swap in better ball physics, better policy (team-specific), or real tracking-derived positions/fatigue.

**Common pitfalls to watch**

* **Columns**: `REQ_COLS` must be present (your batch code coalesces `_x/_y` to base names—good).
* **Units**: If coordinates are normalized but not scaled, distances will be tiny and success fits odd. `fit_match_model` scales if `max_abs ≤ 1.2`.
* **Teams**: `control_field` expects exactly 2 team names present; if not, returns zeros (control has no signal).
* **State mutation across rollouts**: In a purist setup you’d deepcopy the `GameState` each rollout. If you see cross-rollout contamination, switch to a deep copy before the steps loop.

**Tuning levers**

* **Policy entropy**: If actions look too random, smooth counts less (remove +1e-6) or lower `temperature` (if you wire it in sampling).
* **Success slopes**: Make `b_d` more negative (pass/carry) or `b_dist` more negative (shot) to penalize long actions.
* **Pressure threshold**: Change `hazard_here>0.5` decision or scale hazard differently.
* **Half-life**: `half_life_events` controls how fast overall sequence confidence shrinks with K.

---

# 13) End-to-end flow (one seed)

1. Pass a merged row (events+synthetic) + rough rosters into `construct_state_from_row` → `GameState`.
2. Fit the match once via `fit_match_model(df)` → `MatchFit`.
3. Call `simulate_rollouts(state0, fit, cfg)`:

   * Ensures `control_field` and `hazard_field`.
   * For `R` times:

     * For `K` steps:

       * Choose action from `PolicyModel` at `(ix,iy,phase)`.
       * Sample a target `(tx,ty)` and distance.
       * Compute `p_succ` via `SuccessModel`.
       * Draw success, update carrier/ball/possession/time, recompute fields.
       * Append step record and per-step confidence.
     * Multiply confidences, apply horizon decay → `C_seq`.
   * Aggregate patterns and shot rates by step; return summary + sample sequences.
