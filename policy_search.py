# policy_search.py
"""
simple offline 'learning' of a safer blackstart policy.

We treat the policy parameters (alpha_load, beta_fire, max_fire_risk,
high_risk_trip) as knobs. We run many simulations with random settings and
pick the parameters that maximise the aggregate score returned by simulate().
"""

import random
from typing import Dict, Tuple

from simulator import simulate


SCENARIOS = ["A", "B", "D"]  # we train on fire near L2, fire near S2, and two fires


def run_one_policy(
    alpha_load: float,
    beta_fire: float,
    max_fire_risk: float,
    high_risk_trip: float,
    n_trials_per_scenario: int = 5,
) -> Dict[str, float]:
    """
    Evaluate a set of policy parameters over several scenarios / random seeds.
    Returns average metrics and overall score.
    """
    total_score = 0.0
    total_load = 0.0
    total_exposure = 0.0
    runs = 0

    for scenario in SCENARIOS:
        for trial in range(n_trials_per_scenario):
            seed = trial + 1234  # deterministic but different by trial
            result = simulate(
                scenario=scenario,
                max_steps=5,
                spread_speed_true=0.8,
                wind_speed=0.0,
                wind_direction_deg=0.0,
                seed=seed,
                alpha_load=alpha_load,
                beta_fire=beta_fire,
                max_fire_risk=max_fire_risk,
                high_risk_trip=high_risk_trip,
            )
            m = result["metrics"]
            total_score += m["score"]
            total_load += m["total_load_served"]
            total_exposure += m["fire_exposure"]
            runs += 1

    return {
        "avg_score": total_score / runs,
        "avg_load": total_load / runs,
        "avg_exposure": total_exposure / runs,
    }


def random_policy_search(
    n_samples: int = 40,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Sample random policy parameters and keep the best one.
    """
    best_params = None
    best_metrics = None

    for i in range(n_samples):
        # sample policy parameters from reasonable ranges
        alpha_load = random.uniform(0.5, 2.0)
        beta_fire = random.uniform(1.0, 4.0)
        max_fire_risk = random.uniform(0.4, 0.9)
        high_risk_trip = random.uniform(0.7, 0.99)

        metrics = run_one_policy(
            alpha_load=alpha_load,
            beta_fire=beta_fire,
            max_fire_risk=max_fire_risk,
            high_risk_trip=high_risk_trip,
            n_trials_per_scenario=5,
        )

        print(
            f"[{i+1}/{n_samples}] "
            f"alpha={alpha_load:.2f}, beta={beta_fire:.2f}, "
            f"max_risk={max_fire_risk:.2f}, trip={high_risk_trip:.2f} -> "
            f"score={metrics['avg_score']:.2f}, "
            f"load={metrics['avg_load']:.2f}, "
            f"exposure={metrics['avg_exposure']:.4f}"
        )

        if best_metrics is None or metrics["avg_score"] > best_metrics["avg_score"]:
            best_metrics = metrics
            best_params = {
                "alpha_load": alpha_load,
                "beta_fire": beta_fire,
                "max_fire_risk": max_fire_risk,
                "high_risk_trip": high_risk_trip,
            }

    return best_params, best_metrics


if __name__ == "__main__":
    random.seed(0)
    best_params, best_metrics = random_policy_search(n_samples=40)

    print("\n=== BEST POLICY FOUND ===")
    print(best_params)
    print(best_metrics)
    print(
        "\nYou can now plug these parameters into the Streamlit app by passing "
        "them into simulator.simulate(...) instead of the defaults."
    )
