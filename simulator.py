# simulator.py
from grid_model import build_grid, initial_state
import random
import math
from typing import Dict, List, Tuple, Optional


# -------------------------------------------------
# Node coordinates (toy grid geometry)
# -------------------------------------------------
# Layout:
#    G1 — S1 — S2
#          |     |
#         L1    L2
#
NODE_COORDS = {
    "G1": (0.0, 0.0),
    "S1": (1.0, 0.0),
    "S2": (2.0, 0.0),
    "L1": (1.0, -1.0),
    "L2": (2.0, -1.0),
}


# -------------------------------------------------
# Wildfire model (true physics + satellite estimate)
# -------------------------------------------------

def logistic(x: float) -> float:
    if x < -10:
        return 0.0
    if x > 10:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def moving_fire_front(
    base_origin: Tuple[float, float],
    t: int,
    wind_vec: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Fire origin drift over time (super simple):
      origin_t = base_origin + wind_vec * t
    """
    fx0, fy0 = base_origin
    wx, wy = wind_vec
    return fx0 + wx * t, fy0 + wy * t


def update_true_fire(
    G,
    state: Dict,
    t: int,
    fire_origins: Optional[List[Tuple[float, float]]],
    spread_speed: float,
    wind_vec: Tuple[float, float],
) -> None:
    """
    Update TRUE wildfire intensity (hidden ground truth).

    fire_origins: list of (x,y) starting points, or None for 'no wildfire'
    spread_speed: how fast the front grows
    wind_vec:     (wx, wy) drift per time step of the fire front
    """
    if not fire_origins:
        # no wildfire in this scenario
        for n in G.nodes:
            state[n]["true_fire"] = 0.0
        return

    for n in G.nodes:
        x, y = NODE_COORDS.get(n, (0.0, 0.0))

        # multiple fires = take the max intensity from each front
        intensities = []
        for base_origin in fire_origins:
            fx, fy = moving_fire_front(base_origin, t, wind_vec)
            distance = math.dist((x, y), (fx, fy))
            intensities.append(logistic(spread_speed * t - distance))

        state[n]["true_fire"] = max(intensities) if intensities else 0.0


def update_ai_fire(
    G,
    state: Dict,
    noise_std: float = 0.08,
    bias: float = 0.05,
    lag_steps: int = 1,
) -> None:
    """
    AI satellite model: noisy, slightly biased estimate of fire.

    - Uses a *lagged* version of true_fire to mimic satellite latency.
    - Adds Gaussian noise + a small upward bias (safety conservative).
    - Stored in state[n]["fire_risk"].
    """
    for n in G.nodes:
        true_val = state[n].get("true_fire", 0.0)
        lagged = max(0.0, true_val - 0.1 * lag_steps)

        noisy = lagged + bias + random.gauss(0.0, noise_std)
        state[n]["fire_risk"] = min(1.0, max(0.0, noisy))


# -------------------------------------------------
# Sensor + belief updates
# -------------------------------------------------

def update_sensors(G, state, p_fail=0.2):
    """
    Randomly make some sensors fail.
    p_fail: probability per step that a working sensor stops reporting.
    """
    for n in G.nodes:
        # keep generator sensor always on for clarity
        if G.nodes[n]["type"] == "generator":
            continue

        if state[n]["sensor_ok"] and random.random() < p_fail:
            state[n]["sensor_ok"] = False
            state[n]["measurement"] = None


def update_beliefs(G, state):
    """
    Update belief_energized for each node.
    - If sensor works, trust its measurement.
    - If sensor is dead, keep previous belief BUT
      if wildfire risk is extremely high, assume it is unsafe / de-energized.
    """
    for n in G.nodes:
        s = state[n]
        fire_risk = s.get("fire_risk", 0.0)

        if s["sensor_ok"] and s["measurement"] is not None:
            s["belief_energized"] = s["measurement"]
        else:
            # sensor failed -> persistent belief, but override if crazy fire
            if fire_risk > 0.9:
                s["belief_energized"] = False
            # else: keep old belief as-is


# -------------------------------------------------
# Decision / planning (policy parameters live here)
# -------------------------------------------------

def node_score(G, state, node, alpha_load: float, beta_fire: float) -> float:
    """
    How attractive is it to energize this node?

    We want:
      - high load
      - low AI-estimated fire_risk
    """
    load = G.nodes[node].get("load_mw", 0)
    fire_risk = state[node].get("fire_risk", 0.0)
    return alpha_load * load - beta_fire * fire_risk


def choose_next_node(
    G,
    state,
    alpha_load: float,
    beta_fire: float,
    max_fire_risk: float,
) -> Optional[str]:
    """
    'AI decision':
    - Use belief_energized to find frontier nodes.
    - Don't pick neighbors that are already energized.
    - Don't pick neighbors whose fire_risk is too high.
    - Among remaining, choose node with highest node_score.
    """
    candidates = []  # (node, score, load, fire_risk)

    for n in G.nodes:
        if state[n]["belief_energized"]:
            for nbr in G.neighbors(n):
                if state[nbr]["energized"]:
                    continue

                fire_risk = state[nbr].get("fire_risk", 0.0)
                if fire_risk > max_fire_risk:
                    continue

                score = node_score(G, state, nbr, alpha_load, beta_fire)
                load = G.nodes[nbr].get("load_mw", 0)
                candidates.append((nbr, score, load, fire_risk))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    next_node, score, load, fire_risk = candidates[0]
    print(
        f"Decision: choosing {next_node} "
        f"(score={score:.2f}, load={load} MW, fire_risk={fire_risk:.2f})"
    )
    return next_node


# -------------------------------------------------
# Single simulation step
# -------------------------------------------------

def step(
    G,
    state,
    t: int,
    fire_origins: Optional[List[Tuple[float, float]]],
    spread_speed_true: float,
    wind_vec: Tuple[float, float],
    use_wildfire: bool,
    alpha_load: float,
    beta_fire: float,
    max_fire_risk: float,
    high_risk_trip: float,
) -> Optional[str]:
    """
    One simulation step:
      - update true fire (if wildfire is enabled)
      - update AI fire estimate
      - random sensor failures
      - update beliefs
      - choose next node
      - energize it + update measurements + beliefs
      - auto-trip if fire becomes extremely high (high_risk_trip)
    """
    if use_wildfire and fire_origins:
        update_true_fire(G, state, t, fire_origins, spread_speed_true, wind_vec)
        update_ai_fire(G, state)
    else:
        # no wildfire scenario
        for n in G.nodes:
            state[n]["true_fire"] = 0.0
            state[n]["fire_risk"] = 0.0

    update_sensors(G, state, p_fail=0.2)
    update_beliefs(G, state)

    node = choose_next_node(G, state, alpha_load, beta_fire, max_fire_risk)
    if node is None:
        return None

    # ground truth: energize it
    state[node]["energized"] = True
    state[node]["belief_energized"] = True

    if state[node]["sensor_ok"]:
        state[node]["measurement"] = True

    update_beliefs(G, state)

    # auto-trip very high TRUE fire nodes
    for n in G.nodes:
        if state[n]["energized"] and state[n].get("true_fire", 0.0) > high_risk_trip:
            state[n]["energized"] = False
            state[n]["belief_energized"] = False
            if state[n]["sensor_ok"]:
                state[n]["measurement"] = False

    return node


# -------------------------------------------------
# Public API: simulate()  (for Streamlit & learning)
# -------------------------------------------------

def simulate(
    scenario: str,
    max_steps: int = 5,
    spread_speed_true: float = 0.8,
    wind_speed: float = 0.0,
    wind_direction_deg: float = 0.0,
    seed: Optional[int] = None,
    # --- policy parameters (Option 3) ---
    alpha_load: float = 1.0,
    beta_fire: float = 2.0,
    max_fire_risk: float = 0.8,
    high_risk_trip: float = 0.95,
):
    """
    Run one simulation and return structured logs for Streamlit + learning.

    scenario: "A", "B", "C", or "D"
    wind_speed: units per time-step (in grid coordinate space)
    wind_direction_deg: 0 = +x, 90 = +y, etc.
    Policy parameters:
      - alpha_load: weight on load in node_score
      - beta_fire:  weight on fire_risk in node_score
      - max_fire_risk: hard cutoff to ever energize a node
      - high_risk_trip: TRUE fire level at which energized nodes auto-trip
    """
    if seed is not None:
        random.seed(seed)

    G = build_grid()
    state = initial_state(G)

    # ensure 3-layer keys exist
    for n in G.nodes:
        state[n]["true_fire"] = state[n].get("true_fire", 0.0)
        state[n]["fire_risk"] = state[n].get("fire_risk", 0.0)

    # Scenario mapping -> list of fire origins
    if scenario == "A":
        fire_origins = [(2.5, -1.0)]          # Fire near L2
        use_wildfire = True
    elif scenario == "B":
        fire_origins = [(2.0, 0.5)]           # Fire near S2
        use_wildfire = True
    elif scenario == "D":
        fire_origins = [(2.5, -1.0), (2.0, 0.5)]  # Two fires
        use_wildfire = True
    else:  # "C" or anything else: no wildfire
        fire_origins = []
        use_wildfire = False

    # wind vector from polar inputs
    theta = math.radians(wind_direction_deg)
    wind_vec = (wind_speed * math.cos(theta), wind_speed * math.sin(theta))

    decision_log = []

    # --- metrics for learning (Option 3) ---
    total_load_served = 0.0        # sum over time of energized load
    fire_exposure = 0.0            # sum over time of load * true_fire

    # Run simulation
    for t in range(max_steps):
        chosen = step(
            G,
            state,
            t,
            fire_origins=fire_origins,
            spread_speed_true=spread_speed_true,
            wind_vec=wind_vec,
            use_wildfire=use_wildfire,
            alpha_load=alpha_load,
            beta_fire=beta_fire,
            max_fire_risk=max_fire_risk,
            high_risk_trip=high_risk_trip,
        )

        energized_nodes = sorted(
            [n for n, s in state.items() if s["energized"]]
        )

        # metrics at this step
        step_load = 0.0
        step_exposure = 0.0
        for n in G.nodes:
            load = G.nodes[n].get("load_mw", 0.0)
            if state[n]["energized"]:
                step_load += load
                step_exposure += load * state[n].get("true_fire", 0.0)

        total_load_served += step_load
        fire_exposure += step_exposure

        decision_log.append(
            {
                "t": t,
                "chosen_node": chosen if chosen is not None else "None",
                "energized_nodes": ", ".join(energized_nodes),
            }
        )

        if chosen is None:
            break

    # Final summaries
    final_true_fire = []
    final_ai_fire = []
    final_beliefs = []

    for n in G.nodes:
        final_true_fire.append(
            {"node": n, "true_fire": round(state[n].get("true_fire", 0.0), 3)}
        )
        final_ai_fire.append(
            {"node": n, "ai_fire": round(state[n].get("fire_risk", 0.0), 3)}
        )
        final_beliefs.append(
            {
                "node": n,
                "energized": state[n]["energized"],
                "belief_energized": state[n]["belief_energized"],
            }
        )

    # simple scalar objective for learning:
    # serve load but penalize placing load in high fire.
    exposure_penalty = 5.0
    score = total_load_served - exposure_penalty * fire_exposure

    return {
        "decision_log": decision_log,
        "final_true_fire": final_true_fire,
        "final_ai_fire": final_ai_fire,
        "final_beliefs": final_beliefs,
        "node_coords": [
            {"node": n, "x": NODE_COORDS[n][0], "y": NODE_COORDS[n][1]}
            for n in G.nodes
        ],
        "metrics": {
            "total_load_served": total_load_served,
            "fire_exposure": fire_exposure,
            "score": score,
            "alpha_load": alpha_load,
            "beta_fire": beta_fire,
            "max_fire_risk": max_fire_risk,
            "high_risk_trip": high_risk_trip,
        },
    }


# -------------------------------------------------
# CLI demo (keeps your existing behaviour)
# -------------------------------------------------

def run_cli_demo():
    scenarios = {
        "Scenario A: Fire near L2 (load-side)": "A",
        "Scenario B: Fire near S2 (substation hub)": "B",
        "Scenario C: No wildfire (baseline)": "C",
    }

    for label, code in scenarios.items():
        print(f"\n=== {label} ===")
        result = simulate(
            code,
            max_steps=5,
            spread_speed_true=0.8,
            wind_speed=0.0,
            wind_direction_deg=0.0,
            seed=42,
        )

        for row in result["decision_log"]:
            print(row)
        print("Final TRUE fire:", result["final_true_fire"])
        print("Final AI fire_risk:", result["final_ai_fire"])
        print("Final beliefs:", result["final_beliefs"])
        print("Metrics:", result["metrics"])
        print("=" * 60)


if __name__ == "__main__":
    run_cli_demo()
