# app.py (or streamlit_app.py)
import streamlit as st
import pandas as pd
import altair as alt

from simulator import simulate  # import the new API

st.set_page_config(
    page_title="Wildfire-Aware Blackstart Simulator",
    layout="wide",
)

st.sidebar.title("Simulation settings")

scenario_label = st.sidebar.selectbox(
    "Scenario",
    [
        "Scenario A: Fire near L2 (load-side)",
        "Scenario B: Fire near S2 (substation hub)",
        "Scenario C: No wildfire (baseline)",
        "Scenario D: Two fires (L2 + S2)",  # new multi-fire option
    ],
)

# Map pretty labels to internal codes
SCENARIO_CODE = {
    "Scenario A: Fire near L2 (load-side)": "A",
    "Scenario B: Fire near S2 (substation hub)": "B",
    "Scenario C: No wildfire (baseline)": "C",
    "Scenario D: Two fires (L2 + S2)": "D",
}

max_steps = st.sidebar.slider("Max time steps", 1, 10, 5)

spread_speed = st.sidebar.slider("True fire spread speed", 0.2, 2.0, 0.8, 0.1)

with st.sidebar.expander("Wind (advanced)", expanded=False):
    wind_speed = st.slider("Wind speed", 0.0, 2.0, 0.0, 0.1)
    wind_dir = st.slider("Wind direction (deg, 0 = +x)", 0, 360, 0)

if st.sidebar.button("Run simulation"):
    code = SCENARIO_CODE[scenario_label]

    result = simulate(
        scenario=code,
        max_steps=max_steps,
        spread_speed_true=spread_speed,
        wind_speed=wind_speed,
        wind_direction_deg=wind_dir,
        seed=42,
    )

    st.title("🔥 Wildfire-Aware Blackstart Simulator")

    st.markdown(
        """
        This app shows how an AI-assisted control agent restores a simple 5-node
        grid under different wildfire scenarios, using:

        • **True fire** (hidden physics layer)  
        • **AI fire_risk** (satellite-style estimate with noise + lag)  
        • **Beliefs** about which nodes are energized, with failing sensors
        """
    )

    # ---------- Decision log ----------
    st.subheader("Decision log")
    df_decisions = pd.DataFrame(result["decision_log"])
    st.dataframe(df_decisions, use_container_width=True)

    # ---------- Final fire risk table + bar chart ----------
    st.subheader("Final-step fire risk: True vs AI estimate")

    df_true = pd.DataFrame(result["final_true_fire"])
    df_ai = pd.DataFrame(result["final_ai_fire"])
    df_fire = df_true.merge(df_ai, on="node")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Table")
        st.dataframe(df_fire, use_container_width=True)

    with col2:
        st.caption("Bar chart")
        fire_long = df_fire.melt("node", var_name="type", value_name="value")
        chart = (
            alt.Chart(fire_long)
            .mark_bar()
            .encode(
                x="node:N",
                y="value:Q",
                color="type:N",
                column=alt.Column("type:N", header=alt.Header(title=None)),
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # ---------- Final beliefs vs energized ----------
    st.subheader("Final-step beliefs vs actual energized state")
    df_beliefs = pd.DataFrame(result["final_beliefs"])
    st.dataframe(df_beliefs, use_container_width=True)

    # ---------- Optional: show node geometry ----------
    with st.expander("Show node coordinates (toy geometry)"):
        st.dataframe(pd.DataFrame(result["node_coords"]), use_container_width=True)

else:
    st.title("🔥 Wildfire-Aware Blackstart Simulator")
    st.markdown(
        "Choose a scenario and click **Run simulation** in the sidebar."
    )
