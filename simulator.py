from grid_model import build_grid, initial_state

def choose_next_node(G, state):
    candidates = []

    for n in G.nodes:
        if state[n]["energized"]:
            for nbr in G.neighbors(n):
                if not state[nbr]["energized"]:
                    load = G.nodes[nbr].get("load_mw", 0)
                    candidates.append((nbr, load))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def step(G, state):
    node = choose_next_node(G, state)
    if node is None:
        return None

    state[node]["energized"] = True
    state[node]["belief_energized"] = True
    return node

def run_demo():
    G = build_grid()
    state = initial_state(G)

    print("Initial state:", state)
    while True:
        chosen = step(G, state)
        if chosen is None:
            print("No more nodes to energize.")
            break
        print(f"Energized {chosen}. Current state:")
        print(state)

if __name__ == "__main__":
    run_demo()
