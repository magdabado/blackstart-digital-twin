import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

from grid_model import build_grid, initial_state
from simulator import step

st.set_page_config(page_title="Blackstart Digital Twin", layout="wide")

# --- Session state so our grid/state persist between button clicks ---
if "G" not in st.session_state:
    st.session_state.G = build_grid()
if "state" not in st.session_state:
    st.session_state.state = initial_state(st.session_state.G)

G = st.session_state.G
state = st.session_state.state

st.title("Blackstart Digital Twin (MVP)")
st.write("Green = energized, Gray = off")

# --- Draw the grid using NetworkX + matplotlib ---
pos = nx.spring_layout(G, seed=42)

node_colors = [
    "lightgreen" if state[n]["energized"] else "lightgray"
    for n in G.nodes
]

fig, ax = plt.subplots()
nx.draw_networkx(G, pos, with_labels=True, node_color=node_colors, ax=ax)
ax.axis("off")
st.pyplot(fig)

# --- Controls ---
col1, col2 = st.columns(2)

with col1:
    if st.button("Run one step"):
        chosen = step(G, state)
        if chosen is None:
            st.info("No more nodes can be energized.")
        else:
            st.success(f"Energized {chosen}")

with col2:
    if st.button("Reset"):
        st.session_state.state = initial_state(G)
        st.experimental_rerun()
