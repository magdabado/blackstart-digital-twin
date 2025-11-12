import networkx as nx

def build_grid():
    """
    Create a tiny example grid:
    - G1: generator (blackstart-capable)
    - S1, S2: substations
    - L1, L2: load buses
    """
    G = nx.Graph()

    # Add nodes
    G.add_node("G1", type="generator", load_mw=0)
    G.add_node("S1", type="substation", load_mw=0)
    G.add_node("S2", type="substation", load_mw=0)
    G.add_node("L1", type="load", load_mw=10)
    G.add_node("L2", type="load", load_mw=15)

    # Lines with capacities
    G.add_edge("G1", "S1", capacity_mw=30)
    G.add_edge("S1", "S2", capacity_mw=20)
    G.add_edge("S1", "L1", capacity_mw=15)
    G.add_edge("S2", "L2", capacity_mw=15)

    return G

def initial_state(G):
    state = {}
    for n, data in G.nodes(data=True):
        energized = (data["type"] == "generator")

        state[n] = {
            "energized": energized,
            "sensor_ok": True,
            "belief_energized": energized
        }
    return state
