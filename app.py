"""
app.py
Cloud Resource Allocation — Game Theory Simulation Dashboard

A minimal, modern, premium-grade dashboard modeling strategic cloud resource allocation,
proportional fair sharing, and pure-strategy Nash equilibria.

Strictly adheres to the designated color palette:
  --deep-space-blue:        #003049
  --deep-space-blue-light:  #00679F
  --deep-space-blue-bright: #00A0F7
  --flag-red:               #D62828
  --vivid-tangerine:        #F77F00
  --sunflower-gold:         #FCBF49
  --vanilla-custard:        #EAE2B7
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from game_theory import (
    STRATEGY_LOW,
    STRATEGY_HIGH,
    STRATEGIES,
    compute_allocation,
    compute_payoff,
    generate_parametric_payoff_matrix,
    analyze_game,
    validate_inputs,
)

# Page configuration
st.set_page_config(
    page_title="Cloud Resource Allocation",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color tokens
PALETTE = {
    "deep_space_blue": "#003049",
    "deep_space_blue_light": "#00679F",
    "deep_space_blue_bright": "#00A0F7",
    "flag_red": "#D62828",
    "vivid_tangerine": "#F77F00",
    "sunflower_gold": "#FCBF49",
    "vanilla_custard": "#EAE2B7",
}

# Scoped custom styling for a restrained, minimal SaaS aesthetic
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #003049;
    }

    /* Shift content up: eliminate Streamlit's default 6rem top padding */
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 0.6rem !important;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
        max-width: 100% !important;
    }

    /* Tighten vertical gaps between Streamlit blocks */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.4rem !important;
    }

    /* Minimal Header */
    .dashboard-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 6px;
        margin-bottom: 8px;
        border-bottom: 1px solid rgba(0, 48, 73, 0.12);
    }
    .header-left {
        display: flex;
        align-items: baseline;
        gap: 12px;
    }
    .header-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #003049;
        margin: 0;
        line-height: 1.1;
    }
    .header-subtitle {
        font-size: 0.82rem;
        color: #003049;
        opacity: 0.72;
        margin: 0;
    }
    .status-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background-color: #EAE2B7;
        color: #003049;
        border: 1px solid rgba(0, 48, 73, 0.15);
    }

    /* Compact Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid rgba(0, 48, 73, 0.12);
        border-radius: 6px;
        padding: 8px 12px;
        transition: border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #00679F;
    }
    .metric-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #003049;
        opacity: 0.65;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.50rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #003049;
        line-height: 1.1;
        margin-bottom: 2px;
    }
    .metric-caption {
        font-size: 0.72rem;
        color: #003049;
        opacity: 0.75;
    }

    /* Panel Headers */
    .panel-heading {
        font-size: 0.94rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #003049;
        margin: 0 0 2px 0;
    }
    .panel-subheading {
        font-size: 0.74rem;
        color: #003049;
        opacity: 0.70;
        margin: 0 0 6px 0;
    }

    /* Compact Strategic Payoff Table */
    .payoff-grid {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border: 1px solid rgba(0, 48, 73, 0.14);
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 8px;
        font-size: 0.82rem;
    }
    .payoff-grid th {
        background: #EAE2B7;
        color: #003049;
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 6px 8px;
        border-bottom: 1px solid rgba(0, 48, 73, 0.14);
        border-right: 1px solid rgba(0, 48, 73, 0.14);
        text-align: center;
    }
    .payoff-grid td {
        padding: 6px 6px;
        border-bottom: 1px solid rgba(0, 48, 73, 0.1);
        border-right: 1px solid rgba(0, 48, 73, 0.1);
        text-align: center;
        background: #ffffff;
        vertical-align: middle;
        position: relative;
    }
    .payoff-grid tr:last-child td {
        border-bottom: none;
    }
    .payoff-grid td:last-child, .payoff-grid th:last-child {
        border-right: none;
    }
    .payoff-tuple {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #003049;
    }
    .cell-nash {
        background: rgba(0, 160, 247, 0.08) !important;
        outline: 2px solid #00A0F7 !important;
        outline-offset: -2px;
    }
    .cell-active {
        background: rgba(234, 226, 183, 0.25) !important;
    }
    .nash-badge {
        display: inline-block;
        background: #003049;
        color: #ffffff;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 2px 6px;
        border-radius: 3px;
        text-transform: uppercase;
        margin-top: 3px;
    }
    .active-badge {
        display: inline-block;
        background: #00679F;
        color: #ffffff;
        font-size: 0.60rem;
        font-weight: 600;
        padding: 1px 5px;
        border-radius: 3px;
        margin-top: 3px;
    }
    .br-tag {
        display: inline-block;
        font-size: 0.60rem;
        font-weight: 600;
        padding: 1px 4px;
        border-radius: 2px;
        background: #EAE2B7;
        color: #003049;
        margin: 2px 1px 0 1px;
    }

    /* Understated Footer */
    .dashboard-footer {
        border-top: 1px solid rgba(0, 48, 73, 0.1);
        padding-top: 6px;
        margin-top: 8px;
        font-size: 0.72rem;
        color: #003049;
        opacity: 0.65;
        text-align: center;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR: COMPACT, ORGANIZED CONFIGURATION PANEL
# ==============================================================================
st.sidebar.markdown(f"""
<div style="margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid rgba(0, 48, 73, 0.1);">
    <span class="status-pill">Control Panel</span>
    <h3 style="font-size: 1.1rem; font-weight: 700; color: {PALETTE['deep_space_blue']}; margin: 6px 0 0 0;">Configuration</h3>
</div>
""", unsafe_allow_html=True)

# Group 1: Infrastructure
st.sidebar.markdown(f"**1. Infrastructure**")
capacity = st.sidebar.number_input(
    "Total CPU Capacity (units)",
    min_value=1.0,
    max_value=1000.0,
    value=100.0,
    step=10.0,
    help="Finite shared physical compute capacity."
)

# Group 2: Applications & Strategy Profile
st.sidebar.markdown(f"**2. Applications**")
strat_col1, strat_col2 = st.sidebar.columns(2)
with strat_col1:
    strategy_a = st.selectbox(
        "App A Strategy",
        STRATEGIES,
        index=1,
        help="Strategy selected for Application A in the live scenario."
    )
with strat_col2:
    strategy_b = st.selectbox(
        "App B Strategy",
        STRATEGIES,
        index=1,
        help="Strategy selected for Application B in the live scenario."
    )

# Advanced Demands hidden by default in a clean expander
with st.sidebar.expander("Application Demands (Low / High)", expanded=False):
    st.markdown("<p style='font-size: 0.78rem; opacity: 0.75;'>Configure CPU requested under each strategy:</p>", unsafe_allow_html=True)
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.caption("App A Demands")
        a_low = st.number_input("Low (A)", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
        a_high = st.number_input("High (A)", min_value=0.0, max_value=500.0, value=80.0, step=5.0)
    with d_col2:
        st.caption("App B Demands")
        b_low = st.number_input("Low (B)", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
        b_high = st.number_input("High (B)", min_value=0.0, max_value=500.0, value=80.0, step=5.0)

# Group 3: Game Parameters (Hidden in clean expander by default)
st.sidebar.markdown(f"**3. Game Parameters**")
with st.sidebar.expander("Payoff Model & Utility Formulation", expanded=False):
    payoff_mode = st.radio(
        "Payoff Engine",
        ["Resource-Derived (Parametric)", "Direct Matrix (Canonical / Manual)"],
        index=0,
        help="Derive payoffs dynamically from costs & congestion, or directly input cell utilities."
    )

    if payoff_mode == "Resource-Derived (Parametric)":
        preset_choice = st.selectbox(
            "Scenario Preset",
            [
                "Tragedy of the Commons (Prisoner's Dilemma)",
                "Resource Abundance (Sufficient Capacity)",
                "High Congestion Penalty (Severe Overload)",
                "Custom Parameters",
            ],
            index=0
        )

        if preset_choice == "Resource Abundance (Sufficient Capacity)":
            v_init, c_init, g_init, s_init = 0.20, 0.01, 0.00, 0.01
        elif preset_choice == "High Congestion Penalty (Severe Overload)":
            v_init, c_init, g_init, s_init = 0.22, 0.03, 0.08, 0.12
        else:
            v_init, c_init, g_init, s_init = 0.20, 0.01, 0.02, 0.04

        param_val = st.number_input("Value per CPU Unit (v)", min_value=0.01, max_value=2.0, value=v_init, step=0.01)
        param_cost = st.number_input("Cost per CPU Unit (c)", min_value=0.00, max_value=1.0, value=c_init, step=0.005, format="%.3f")
        param_cong = st.number_input("Congestion Penalty (γ)", min_value=0.00, max_value=1.0, value=g_init, step=0.005, format="%.3f")
        param_short = st.number_input("Shortfall Penalty (δ)", min_value=0.00, max_value=1.0, value=s_init, step=0.01)

        matrix = generate_parametric_payoff_matrix(
            capacity=capacity,
            a_low_demand=a_low,
            a_high_demand=a_high,
            b_low_demand=b_low,
            b_high_demand=b_high,
            value_per_unit=param_val,
            cost_per_unit=param_cost,
            congestion_penalty_coeff=param_cong,
            shortfall_penalty_coeff=param_short,
        )
    else:
        matrix_preset = st.selectbox(
            "Matrix Preset",
            ["Canonical Prompt (5,5 / 2,7 / 7,2 / 3,3)", "Coordination Game", "Custom Matrix"],
            index=0
        )
        if matrix_preset == "Coordination Game":
            m_ll, m_lh, m_hl, m_hh = (6.0, 6.0), (1.0, 1.0), (1.0, 1.0), (4.0, 4.0)
        else:
            m_ll, m_lh, m_hl, m_hh = (5.0, 5.0), (2.0, 7.0), (7.0, 2.0), (3.0, 3.0)

        c1, c2 = st.columns(2)
        with c1:
            ua_ll = st.number_input("A: Low, B: Low | UA", value=m_ll[0], step=0.5)
            ua_lh = st.number_input("A: Low, B: High | UA", value=m_lh[0], step=0.5)
            ua_hl = st.number_input("A: High, B: Low | UA", value=m_hl[0], step=0.5)
            ua_hh = st.number_input("A: High, B: High | UA", value=m_hh[0], step=0.5)
        with c2:
            ub_ll = st.number_input("A: Low, B: Low | UB", value=m_ll[1], step=0.5)
            ub_lh = st.number_input("A: Low, B: High | UB", value=m_lh[1], step=0.5)
            ub_hl = st.number_input("A: High, B: Low | UB", value=m_hl[1], step=0.5)
            ub_hh = st.number_input("A: High, B: High | UB", value=m_hh[1], step=0.5)

        matrix = {
            (STRATEGY_LOW, STRATEGY_LOW): (ua_ll, ub_ll),
            (STRATEGY_LOW, STRATEGY_HIGH): (ua_lh, ub_lh),
            (STRATEGY_HIGH, STRATEGY_LOW): (ua_hl, ub_hl),
            (STRATEGY_HIGH, STRATEGY_HIGH): (ua_hh, ub_hh),
        }

# Validation
validation_errors = validate_inputs(capacity, a_low, a_high, b_low, b_high)
if validation_errors:
    for err in validation_errors:
        st.error(f"Configuration Error: {err}")
    st.stop()


# ==============================================================================
# COMPUTATION (CORE LOGIC PRESERVED)
# ==============================================================================
selected_demands = {
    "A": a_low if strategy_a == STRATEGY_LOW else a_high,
    "B": b_low if strategy_b == STRATEGY_LOW else b_high,
}
active_alloc = compute_allocation(capacity, selected_demands["A"], selected_demands["B"])
cells_analysis, equilibria, social_optima = analyze_game(matrix)
active_cell = cells_analysis[(strategy_a, strategy_b)]
is_active_nash = active_cell.is_nash_equilibrium


# ==============================================================================
# MINIMAL HEADER
# ==============================================================================
st.markdown(f"""
<div class="dashboard-header">
    <div class="header-left">
        <h1 class="header-title">Cloud Resource Allocation</h1>
        <p class="header-subtitle">A game theory simulation for smarter cloud computing decisions.</p>
    </div>
    <div>
        <span class="status-pill">SIMULATION</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: FOUR CLEAN PRIMARY METRIC CARDS
# ==============================================================================
col_m1, col_m2, col_m3, col_m4 = st.columns(4, gap="small")

with col_m1:
    # CPU Utilization with subtle color cue
    util_caption = (
        f"<span style='color: {PALETTE['flag_red']}; font-weight: 600;'>Throttled (At Capacity)</span>"
        if active_alloc.is_throttled
        else f"<span style='color: {PALETTE['deep_space_blue_light']};'>Normal (Under Capacity)</span>"
    )
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">CPU Utilization</div>
        <div class="metric-value">{active_alloc.utilization_pct:.1f}%</div>
        <div class="metric-caption">{util_caption}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">CPU Allocated</div>
        <div class="metric-value">{active_alloc.total_allocated:.1f} <span style="font-size: 0.95rem; font-weight: 500; opacity: 0.6;">units</span></div>
        <div class="metric-caption">of {capacity:.0f} available units</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    rem_caption = "headroom available" if active_alloc.remaining_capacity > 0 else "zero buffer remaining"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Remaining Capacity</div>
        <div class="metric-value">{active_alloc.remaining_capacity:.1f} <span style="font-size: 0.95rem; font-weight: 500; opacity: 0.6;">units</span></div>
        <div class="metric-caption">{rem_caption}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    max_welfare = max(c.social_welfare for c in cells_analysis.values())
    welfare_status = (
        f"<span style='color: {PALETTE['deep_space_blue_light']}; font-weight: 600;'>Social Optimum</span>"
        if abs(active_cell.social_welfare - max_welfare) < 1e-4
        else f"<span>{active_cell.social_welfare - max_welfare:+.1f} vs Optimum ({max_welfare:.1f})</span>"
    )
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Combined Payoff</div>
        <div class="metric-value">{active_cell.social_welfare:.1f}</div>
        <div class="metric-caption">{welfare_status}</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: CENTRAL VISUALIZATION & STRATEGIC ANALYSIS
# ==============================================================================
left_panel, right_panel = st.columns([1.15, 0.85], gap="small")

# ------------------------------------------------------------------------------
# LEFT: PRIMARY VISUALIZATION (HORIZONTAL BAR CHART)
# ------------------------------------------------------------------------------
with left_panel:
    with st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 4px;">
            <h2 class="panel-heading">Resource Allocation</h2>
            <p class="panel-subheading">Comparing requested CPU demand against actual allocated shares under proportional fair sharing.</p>
        </div>
        """, unsafe_allow_html=True)

        # Clean Horizontal Bar Chart
        fig, ax = plt.subplots(figsize=(5.6, 2.05), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        app_labels = ["Application B", "Application A"]
        y_pos = np.arange(len(app_labels))
        bar_height = 0.26

        req_data = [active_alloc.demand_b, active_alloc.demand_a]
        alloc_data = [active_alloc.allocated_b, active_alloc.allocated_a]

        # Horizontal Bars:
        # Requested: Deep Space Blue Light (#00679F) with subtle border
        # Allocated: Deep Space Blue (#003049) solid
        b_req = ax.barh(y_pos + bar_height/2, req_data, bar_height, label="Requested Demand",
                        color=PALETTE["deep_space_blue_light"], edgecolor=PALETTE["deep_space_blue_light"], linewidth=0.8)
        b_alloc = ax.barh(y_pos - bar_height/2, alloc_data, bar_height, label="Allocated CPU",
                          color=PALETTE["deep_space_blue"], edgecolor=PALETTE["deep_space_blue"], linewidth=0.8)

        # Data labels on bars
        for i, bar in enumerate(b_req):
            w = bar.get_width()
            ax.annotate(f"{w:.0f} u",
                        xy=(w, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=7.8, color=PALETTE["deep_space_blue_light"], fontweight="600")

        for i, bar in enumerate(b_alloc):
            w = bar.get_width()
            req_val = req_data[i]
            alloc_ratio = (w / req_val * 100) if req_val > 0 else 100
            ax.annotate(f"{w:.0f} u ({alloc_ratio:.0f}%)",
                        xy=(w, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=7.8, color=PALETTE["deep_space_blue"], fontweight="700")

        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(app_labels, fontsize=9.0, fontweight="600", color=PALETTE["deep_space_blue"])
        ax.set_xlabel("CPU Units", fontsize=8.0, color=PALETTE["deep_space_blue"], fontweight="600")
        ax.tick_params(axis="x", labelsize=7.8, colors=PALETTE["deep_space_blue"])
        ax.tick_params(axis="y", colors=PALETTE["deep_space_blue"])

        # Capacity reference line if within bounds
        ax.axvline(capacity, color=PALETTE["flag_red"], linestyle="--", linewidth=1.2, label=f"Cluster Capacity ({capacity:.0f} u)")

        # Remove extraneous borders and add understated subtle grid
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(PALETTE["vanilla_custard"])
        ax.spines["bottom"].set_color(PALETTE["vanilla_custard"])
        ax.xaxis.grid(True, linestyle=":", alpha=0.35, color=PALETTE["deep_space_blue"])
        ax.set_axisbelow(True)

        max_x = max(max(req_data), capacity) * 1.30
        ax.set_xlim(0, max_x)

        ax.legend(loc="upper right", frameon=False, fontsize=7.6)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Understated allocation policy note
        st.markdown(f"""
        <div style="font-size: 0.76rem; color: {PALETTE['deep_space_blue']}; opacity: 0.8; margin-top: 4px; padding: 6px 10px; background: rgba(234, 226, 183, 0.35); border-radius: 4px; border-left: 2px solid {PALETTE['deep_space_blue_light']};">
            <strong>Allocation Policy:</strong> {active_alloc.policy_description}
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# RIGHT: COMPACT STRATEGIC ANALYSIS
# ------------------------------------------------------------------------------
with right_panel:
    with st.container(border=True):
        st.markdown(f"""
        <div style="margin-bottom: 8px;">
            <h2 class="panel-heading">Strategic Analysis</h2>
            <p class="panel-subheading">Payoff matrix entries represent <strong>(App A, App B)</strong> utilities. Evaluated in pure strategies.</p>
        </div>
        """, unsafe_allow_html=True)

        # 2x2 Matrix HTML
        def render_cell(sa, sb):
            cell = cells_analysis[(sa, sb)]
            is_active = (sa == strategy_a and sb == strategy_b)
            classes = []
            if cell.is_nash_equilibrium:
                classes.append("cell-nash")
            if is_active:
                classes.append("cell-active")

            html = f"""<td class="{' '.join(classes)}">"""
            html += f"""<div class="payoff-tuple">({cell.payoff_a:.1f}, {cell.payoff_b:.1f})</div>"""

            # Tags
            tags = []
            if cell.is_a_best_response:
                tags.append(f"""<span class="br-tag">A: BR</span>""")
            if cell.is_b_best_response:
                tags.append(f"""<span class="br-tag">B: BR</span>""")
            if tags:
                html += f"""<div>{''.join(tags)}</div>"""

            if cell.is_nash_equilibrium:
                html += f"""<div><span class="nash-badge">★ Nash Eq</span></div>"""
            elif is_active:
                html += f"""<div><span class="active-badge">Active</span></div>"""

            html += "</td>"
            return html

        table_code = f"""
        <table class="payoff-grid">
            <thead>
                <tr>
                    <th style="width: 28%;">App A \\ B</th>
                    <th style="width: 36%;">B: Low</th>
                    <th style="width: 36%;">B: High</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <th>A: Low</th>
                    {render_cell(STRATEGY_LOW, STRATEGY_LOW)}
                    {render_cell(STRATEGY_LOW, STRATEGY_HIGH)}
                </tr>
                <tr>
                    <th>A: High</th>
                    {render_cell(STRATEGY_HIGH, STRATEGY_LOW)}
                    {render_cell(STRATEGY_HIGH, STRATEGY_HIGH)}
                </tr>
            </tbody>
        </table>
        """
        st.markdown(table_code, unsafe_allow_html=True)

        # Equilibrium Status Indicator
        if len(equilibria) == 0:
            status_color = PALETTE["flag_red"]
            status_msg = "No Pure-Strategy Nash Equilibrium exists (cyclic dynamics)."
        else:
            eq_str = ", ".join([f"({sa}, {sb})" for sa, sb in equilibria])
            status_color = PALETTE["deep_space_blue"]
            status_msg = f"Pure-Strategy Nash Equilibrium: <strong>{eq_str}</strong>"

        st.markdown(f"""
        <div style="font-size: 0.78rem; padding: 7px 10px; border-radius: 4px; background: #faf9f5; border: 1px solid rgba(0, 48, 73, 0.12); margin-bottom: 10px;">
            <span style="font-weight: 600; color: {status_color};">Status:</span> {status_msg}
        </div>
        """, unsafe_allow_html=True)

        # Expander for Deep Theoretical Explanation (Hidden by default to reduce visual clutter)
        with st.expander("How the equilibrium is calculated", expanded=False):
            st.markdown(f"""
            <div style="font-size: 0.78rem; line-height: 1.45; color: {PALETTE['deep_space_blue']};">
                <strong>1. Best Response Computation:</strong>
                <ul>
                    <li>Against B(Low): App A compares payoffs and prefers <code>{', '.join([s for s in STRATEGIES if cells_analysis[(s, STRATEGY_LOW)].is_a_best_response])}</code>.</li>
                    <li>Against B(High): App A compares payoffs and prefers <code>{', '.join([s for s in STRATEGIES if cells_analysis[(s, STRATEGY_HIGH)].is_a_best_response])}</code>.</li>
                    <li>Against A(Low): App B compares payoffs and prefers <code>{', '.join([s for s in STRATEGIES if cells_analysis[(STRATEGY_LOW, s)].is_b_best_response])}</code>.</li>
                    <li>Against A(High): App B compares payoffs and prefers <code>{', '.join([s for s in STRATEGIES if cells_analysis[(STRATEGY_HIGH, s)].is_b_best_response])}</code>.</li>
                </ul>
                <strong>2. Nash Equilibrium Condition:</strong><br>
                A cell is a Nash Equilibrium if both players are simultaneously playing their best response. Neither player has a unilateral incentive to deviate.<br><br>
                <strong>3. Tragedy of the Commons (Price of Anarchy):</strong><br>
                When both choose <em>High Demand</em>, the cluster experiences congestion, resulting in lower combined welfare ({cells_analysis[(STRATEGY_HIGH, STRATEGY_HIGH)].social_welfare:.1f}) compared to mutual restraint at <em>Low Demand</em> ({cells_analysis[(STRATEGY_LOW, STRATEGY_LOW)].social_welfare:.1f}).
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# MINIMAL ACADEMIC FOOTER
# ==============================================================================
st.markdown(f"""
<div class="dashboard-footer">
    Cloud Resource Allocation — Game Theory Simulation Tool &nbsp;•&nbsp; Proportional Fair Sharing & Strategic Decision-Making
</div>
""", unsafe_allow_html=True)
