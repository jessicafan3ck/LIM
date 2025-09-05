# Synthetic Data Card

## A) Player Kinematics & Physiology

**Recap on available inputs**

* **Physicals:** `Max Speed (km/h)`, `# Sprints`, `# Speed Runs`, `Total Duration (min)`
* **Events:** timestamps (`match_run_time_in_ms`, `event_end_time_in_ms`), `event`, `event_type`, `pressure`, positions (`x_location_*`, `y_location_*`), `to_player_*`, `team_name`, `from_player_*`

**We synthesize:** $v_i^{\max}, a_i^{\max}, d_i^{\max}, \tau_i^{\text{react}}, \tau_i^{\text{turn}}, f_i(t)$ **(+ GK extras)**

### Speed cap $v_i^{\max}$

* From physicals: $v_i^{\max} = \frac{\text{Max Speed (km/h)}}{3.6}$ (m/s).
* If missing: **team median** of available players; fallback **8.6 m/s**.

### Accel/Decel caps $a_i^{\max}, d_i^{\max}$ (tied to physicals)

* Estimate time to 85% top speed from sprint proxy:

  $$
  t85_i = \text{clamp}\big(2.2 - 0.01 \times (\#\text{Sprints}),\ 1.6,\ 2.2\big)\ \text{[s]}
  $$
* Acceleration cap:

  $$
  a_i^{\max} = 0.85 \cdot \frac{v_i^{\max}}{t85_i}
  $$
* Braking cap:

  $$
  d_i^{\max} = 1.25 \cdot a_i^{\max} \quad \text{(braking > accel)}
  $$
* Biomech clamps: $a_i^{\max} \le 6.5\ \text{m/s}^2$, $d_i^{\max} \le 8.0\ \text{m/s}^2$.

### Base latencies $\tau_i^{\text{react}}, \tau_i^{\text{turn}}$

* Role-agnostic bases: $\tau_i^{\text{react}}, \tau_i^{\text{turn}} \in [0.20, 0.40]$ s.
* Personalize with sprint proxy (more sprints → smaller latencies):

  $$
  \tau_i^{\text{react}} = \text{clamp}\big(0.26 - 0.002\cdot \min(20,\#\text{Sprints}),\ 0.18,\ 0.30\big)
  $$

  $$
  \tau_i^{\text{turn}} = \text{clamp}\big(0.34 - 0.002\cdot \min(20,\#\text{Sprints}),\ 0.26,\ 0.40\big)
  $$
* If GK detected (see below), add **+0.02–0.04 s**.

### Fatigue multiplier $f_i(t)$: event-driven two-reservoir model

* Maintain $F_{\text{fast}}, F_{\text{slow}} \in [0,1]$ with **exponential recovery** between events.
* **Burst detection** $I_{\text{burst}}$ (events only):

  * **CARRY:** start→end displacement $d$ (m): bins **4/8/12 m** → intensities **0.3/0.6/1.0**.
  * **PRESS/DEFENSE:** if `pressure==True` on opponent ball-carrier within 6 s of turnover, add **0.4–0.8** based on closed distance.
  * **QUICK CHAINS:** same player acts ≥2 times within 5 s → **+0.3**.
* **Accumulated load:** $I_{\text{burst\_long}}$ = rolling sum over **30 s**, normalized to **\[0,1]**.
* **Contact proxy** $C$ from event text/tags:

  * Regex on `{FOUL, TACKLE, AERIAL, DUEL, COLLISION}`, or `body_type` includes *Header*, or opponent `opposition_touch` near our action → map to **{0.2, 0.4, 0.7}**.
* **Updates at event time $t_k$:**

  $$
  F_{\text{fast}} \leftarrow \text{clip}\big(F_{\text{fast}} - \alpha_{\text{burst}} I_{\text{burst}} - \alpha_{\text{contact}} C,\, 0,\, 1\big)
  $$

  $$
  F_{\text{slow}} \leftarrow \text{clip}\big(F_{\text{slow}} - \beta_{\text{burst}} I_{\text{burst\_long}} - \beta_{\text{contact}} C,\, 0,\, 1\big)
  $$
* **Recovery** between events with $\tau_{\text{fast}}\approx45$ s, $\tau_{\text{slow}}\approx600$ s; **heavy contact** $(C\ge 0.7)$ multiplies $\tau$’s by **1.5** for \~60 s.
* **Final mix:** $f_i(t) = 0.6\,F_{\text{fast}} + 0.4\,F_{\text{slow}}$.

### Effective kinematics used by control at time $t$

$$
\begin{aligned}
v_i^{\text{eff}}(t) &= v_i^{\max}\,\big(1 - 0.25\,(1 - f_i(t))\big) \\
a_i^{\text{eff}}(t) &= a_i^{\max}\,\big(1 - 0.40\,(1 - f_i(t))\big) \\
d_i^{\text{eff}}(t) &= d_i^{\max}\,\big(1 - 0.20\,(1 - f_i(t))\big) \\
\tau_i^{\text{react,eff}}(t) &= \tau_i^{\text{react}}\,(1 + 0.50\,(1 - f_i(t))) \\
\tau_i^{\text{turn,eff}}(t)  &= \tau_i^{\text{turn}}\,(1 + 0.50\,(1 - f_i(t)))
\end{aligned}
$$

### GK extras (if needed): $\tau_{\text{GK,react}}, r_{\text{GK}}, v_{\text{dive}}$

* **Detection via events:** players with `save_type/save_detail` or repeated `save/goal_kick` events; if ambiguous, pick jersey $\in \{1, 13\}$ if present; else the player with max saves or min outfield actions.
* **Defaults:** $\tau_{\text{GK,react}} = \tau_i^{\text{react}} + 0.02$ s, $r_{\text{GK}} = 3.0$ m, $v_{\text{dive}} = 6.0$ m/s.

---

## B) Ball Flight / Interaction

**Inputs:** pass start/end positions, event timestamps, `event/action_type`, `style/style_additional`, `body_type`.

### Ground-ball decay $\kappa$

* Estimate **per match** from passes with valid `event_end_time_in_ms`:

  * For each **PASS**: distance $d$, travel time $\Delta t = t_{\text{end}}-t_{\text{start}}$.
  * If times missing: approximate $\Delta t \approx d/18$ m/s, capped to **\[0.15, 1.2] s** for initial fit.
* Fit drag: $v(t) = v_0 e^{-\kappa t}$; choose $\kappa_{\text{ground}}$ (\~**1.0–1.6 s⁻¹**) minimizing variance of implied $v_0$.
* If timing unusable: set $\kappa_{\text{ground}} = 1.3\ \text{s}^{-1}$.

### Launch-speed priors $(\mu,\sigma)$ by event type

* **PASS:** $\mu = \text{median}(d/\Delta t)$ (trimmed); $\sigma$ = MAD-scaled.
* **SHOT:** infer $\mu$ from shot distance with cap **28–32 m/s**; $\sigma$ broad.
* **CARRY:** N/A (feet on ball).

### Height class → interception scaling

* **AERIAL** if `body_type` contains *Header* **OR** `style/style_additional` contains {“High”, “Cross”, “Chip”, “Long Ball”}
  → use aerial scaling $s_{\text{aerial}}=1.25$ to opponent interception windows; else ground $s_{\text{ground}}=1.0$.

### First-touch delay $\delta_{\text{ctrl}}$

* Base **0.18 s**; increase under pressure:

  $$
  \delta_{\text{ctrl}} = 0.18 \cdot \big(1 + 0.35 \cdot \mathbf{1}\{\text{pressure}\}\big)
  $$

---

## C) Pitch & Coordinate System

**Inputs:** positions in normalized $[0,1]$ or meters; `half_time`, possibly `direction`.

* **Pitch size:** If positions look $\le 1.2$ in magnitude → treat as **normalized**; set $(105, 68)$ m and scale. Else assume meters and verify from min/max.
* **Attacking direction per half:** If a direction column exists, use it. Else infer by sign of **median $\Delta x$** for **successful passes** in that half.
* **Analysis grid $G$** & weights $w(z)$: choose $n_x=36, n_y=24$, uniform cell weights (configurable).

---

## D) Pitch-Control / Reachability Parameters

**Inputs:** effective kinematics from (A), grid from (C).

* Logistic slope & soft aggregation: **$\beta_{\text{ctrl}}=2.5$**, **$\tau_{\text{ctrl}}=0.35$** (stable defaults; tune later).
* **Interception anisotropy** (forward/lateral/back): multipliers **{1.0, 0.85, 0.75}**; direction = player’s current motion toward attacking goal (use imputed lane orientation).
* **$\epsilon_{\text{TTR}}$ floor & physical caps:** $\text{TTR} \ge 0.05\ \text{s}$; speed/accel never exceed caps in (A).

---

## E) Success Models (Probabilities)

**Inputs:** event geometry, outcome, `pressure`, `line_break_*`, `body_type`, `direction`, `style/_additional`.

Heuristic **logits** from the current match’s **own events** (no external training). Baselines from empirical frequencies by coarse bins; then apply shape constraints.

### Pass success $P_{\text{pass}}(i\to j)$

* **Features:** distance $d$, pass-angle favorability $\cos\theta$ (toward goal or receiver lane), pressure (0/1), height class (ground/aerial), line-break intent (direction).
* **Model:**

  $$
  \text{logit}\ P = \alpha_0 + \alpha_1(-d) + \alpha_2 \cos\theta + \alpha_3(-\text{pressure}) + \alpha_4 \mathbf{1}\{\text{aerial}\} + \alpha_5 \mathbf{1}\{\text{LB align}\}
  $$
* **Init:** $\alpha_0 = \text{logit}(p_{\text{base}})$ where $p_{\text{base}}=\frac{\#\text{completed passes}}{\#\text{passes}}$.
  Set $\alpha_1<0$, $\alpha_2>0$; magnitudes from bin differences; clip $P \in [0.05, 0.98]$.

### Carry success $P_{\text{carry}}$

* **Features:** carry length $d_c$, pressure, opponent density proxy (from G below), central-third indicator.
* **Model:** $\text{logit}\ P = \beta_0 + \beta_1(-d_c) + \beta_2(-\text{pressure}) + \beta_3(-\text{density})$.
  Initialize from carry outcomes.

### Take-on success $P_{\text{takeon}}$

* **Features:** angle vs defender, space ahead (low density), pressure.
* Start **0.45** baseline; subtract with more density/pressure; add if moving to open lane.

### Shot success $P_{\text{shot}}$ (xG-lite)

* **Features:** distance to goal, angle to goal mouth, body type (Header vs Foot), pressure.
* **Model:** $\text{logit}\ P = \gamma_0 + \gamma_1(-\text{dist}_m) + \gamma_2(\text{angle}) + \gamma_3 \mathbf{1}\{\text{Header}\}$ with $\gamma_3<0$.
  Initialize from within-match shots; if sparse, defaults \~**0.1** at 16 m center (taper with distance).
* **Calibration:** identity mapping initially (no Platt/Isotonic); cap extremes; enforce monotone **distance** effect via re-binning if necessary.

---

## F) Decision Policies

**Inputs:** `event_type`, `event`, `sequence_type`, `game_state`, `pressure`, geometry; per-zone counting.

* **Acting policy $\pi(a|s)$** (pass/carry/shot & target choice):
  For each zone (grid cell or third) and phase (open play, set play), compute empirical action distribution from events.
  **Pass target choice:** choose teammate $j$ with probability $\propto \text{softmax}(\eta \cdot \text{score}(i\to j))$ where
  $\text{score} = \text{distance favorability} + \text{angle} + \text{line\_break gain} - \text{pressure penalty}$.
  $\eta$ tuned so entropy $\approx$ observed.
* **Opponent policy $\pi_{\text{opp}}$:** mirror acting team proportions by zone with a **press bias**: in middle third with frequent pressure flags, increase pressing/duel propensity by **+10%**.
* **Risk/exploration $\rho$:** base $\rho=0.6$; modulate with `game_state`:
  losing → **+0.1** after 70’; winning → **−0.1**.

---

## G) Pressure / Hazard Field

**Inputs:** pressure flag on events, `team_units_broken`, `total_team_units`, `line_break_direction/outcome`, imputed opponent density (from positions), `opposition_touch`.

* **Opponent density (events-only imputation):** at each event time, anchor known actors (from/to players) at their $x/y$; distribute teammates/opponents into role-lanes around centroid lines (use `team_shape/team_unit` if present). Smooth in space/time.
* **Hazard $\lambda(z,t)$:**

  $$
  \lambda = \sigma\big(\delta_0 + \delta_1(-\text{dist to nearest opponent}) + \delta_2\mathbf{1}\{\text{central third}\} + \delta_3\mathbf{1}\{\text{pressure nearby}\}\big)
  $$

  Boost in cells where `team_units_broken / total_team_units` is high and `line_break_outcome` indicates success.
  Bound $[0,1]$; apply **2–4 s** moving average to avoid flicker.

---

## H) Uncertainty & Calibration

* No explicit tracking noise; fix hyperparameters.
* **Position noise:** $\sigma_{\text{pos}} = 0.6$ m; **ball noise:** $\sigma_{\text{ball}} = 0.4$ m.
* **Monte Carlo:** $N_{\text{mc}}=50$, horizon $H=4$, discount $\gamma=0.9$.
* **Calibrators** $g_*$: identity (reserved for future).
* **Conformal $\alpha$:** nominal only (no held-out residuals yet).

---

## I) Game / Clock Integration

**Inputs:** event timestamps; physicals `Total Duration (min)`.

* `frame_rate_hz`, timebase alignment → **N/A** (events-only).
* **On-pitch availability map:** for each player, mark **on** from first to last event;
  if physicals duration differs by **>5 min**, extend nearest side proportionally to match `Total Duration (min)`.
  If substitutions encoded in `game_involvement`/`origin`, use those for precise on/off times.

---

## Column-to-Feature Dictionary (quick wiring)

* **Geometry/time:** `x_location_start/end`, `y_location_start/end`, `match_run_time_in_ms`, `event_end_time_in_ms`, `half_time`
* **Entities:** `team_name`, `from_player_name/id`, `to_player_name/id`
* **Context/tags:** `pressure`, `line_break_direction/outcome`, `team_shape`, `team_unit`, `total_team_units`, `sequence_type`, `game_state`, `style`, `style_additional`, `body_type`, `direction`, `opposition_touch`, `action_type`
* **Physicals:** `Max Speed (km/h)`, `# Speed Runs`, `# Sprints`, `Total Duration (min)`

---

## Guardrails (enforced everywhere)

* **Physical coherence:** speeds/accels never exceed caps derived from `Max Speed (km/h)` & sprint proxy; $\text{TTR} \ge 0.05\ \text{s}$.
* **Smoothness:** team control fields via softmin/softmax; hazard via short moving averages.
* **Monotonicity:** longer distances ↓ $P_{\text{pass}}/P_{\text{carry}}/P_{\text{shot}}$; better angles ↑.
* **Probability conservation:** at each cell, $p_T + p_O = 1$ (post-blend).
