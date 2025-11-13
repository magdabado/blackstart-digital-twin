# grid_model.py
import networkx as nx

# -------------------------------------------------
# Toy grid topology + coordinates
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

def build_grid():
    """
    Build a tiny 5-node grid as a NetworkX graph.

    Node attributes:
      - type: "generator" | "substation" | "load"
      - load_mw: demand at that node (only for loads)
      - coord: (x, y) for wildfire geometry
    """

    G = nx.Graph()

    # --- add nodes ---------------------------------------------------
    G.add_node("G1", type="generator", load_mw=0, coord=NODE_COORDS["G1"])
    G.add_node("S1", type="substation", load_mw=0, coord=NODE_COORDS["S1"])
    G.add_node("S2", type="substation", load_mw=0, coord=NODE_COORDS["S2"])

    # loads
    G.add_node("L1", type="load", load_mw=10, coord=NODE_COORDS["L1"])
    G.add_node("L2", type="load", load_mw=15, coord=NODE_COORDS["L2"])

    # --- add lines (edges) -------------------------------------------
    # you could later add line parameters here (impedance, rating, etc.)
    G.add_edge("G1", "S1")
    G.add_edge("S1", "S2")
    G.add_edge("S1", "L1")
    G.add_edge("S2", "L2")

    return G


def initial_state(G):
    """
    Build the per-node state dict used by the simulator.

    For each node n:
      state[n] = {
        "energized": bool   (ground truth)
        "sensor_ok": bool
        "measurement": bool | None
        "belief_energized": bool
        "fire_risk": float          (AI estimate from satellite)
        "true_fire": float          (hidden physical truth)
      }
    """

    state = {}

    for n in G.nodes:
        node_type = G.nodes[n]["type"]

        if node_type == "generator":
            energized = True
            measurement = True
            belief = True
        else:
            energized = False
            measurement = False
            belief = False

        state[n] = {
            "energized": energized,
            "sensor_ok": True,
            "measurement": measurement,
            "belief_energized": belief,
            "fire_risk": 0.0,   # AI's risk estimate (from satellite model)
            "true_fire": 0.0,   # hidden "ground truth" fire intensity
        }

    return state
