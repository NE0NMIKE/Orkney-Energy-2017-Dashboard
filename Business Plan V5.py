import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Kaluza DR Financial Model", layout="wide")
st.title("Kaluza & Wind Farm Shared Business Model")

# --- SIDEBAR: TIMELINE ---
st.sidebar.header("0. Simulation Timeline")
time_resolution = st.sidebar.radio("Time Resolution", ["Monthly", "Yearly"], horizontal=True)

if time_resolution == "Monthly":
    simulation_periods = st.sidebar.slider("Simulation Duration (Months)", min_value=12, max_value=120, value=36,
                                           step=12, help="How many months to run the forecast for.")
    period_label = "Month"
    short_label = "Mo"
    annual_divider = 12
else:
    simulation_periods = st.sidebar.slider("Simulation Duration (Years)", min_value=1, max_value=10, value=3, step=1,
                                           help="How many years to run the forecast for.")
    period_label = "Year"
    short_label = "Yr"
    annual_divider = 1

st.markdown(f"Interactive **{simulation_periods}-{period_label}** Projections for Kaluza and Generation Partners")

# --- SIDEBAR: CORE BUSINESS MODEL ---
st.sidebar.header("1. Core Business Model")
kaluza_rev_share = st.sidebar.slider("Kaluza Revenue Share (%)", min_value=0, max_value=100, value=50, step=5,
                                     help="Percentage of total restored ROC and DNO flexibility revenues retained by Kaluza.") / 100
discount_rate = st.sidebar.slider("Discount Rate / WACC (Annual %)", min_value=1, max_value=20, value=10, step=1,
                                  help="The annual rate used to discount future cash flows to their present value (NPV).") / 100

st.sidebar.header("2. Capex & Installation")
total_install_cost = st.sidebar.number_input("Total Install/Onboarding Cost (£)", value=100,
                                             help="Total upfront cost per household (hardware + acquisition).")
kaluza_capex_split = st.sidebar.slider("Kaluza Share of Install Cost (%)", min_value=0, max_value=100, value=100,
                                       step=10,
                                       help="How much of the onboarding cost Kaluza pays. Reducing this passes costs to the consumer.") / 100

st.sidebar.header("3. Household Growth (Equation)")
growth_type = st.sidebar.selectbox("Growth Curve",
                                   ["Linear (+X per year)", "Flat (No Growth)", "Exponential (+X% per year)"])
initial_hh = st.sidebar.number_input("Initial Households (Starting Base)", value=3000)
growth_rate = st.sidebar.number_input("Annual Growth Rate (X)", value=1200.0)
max_hh = st.sidebar.number_input("Maximum Households (Cap)", value=10000, step=500,
                                 help="The absolute maximum number of households the programme can support. Growth will flatline once this is hit.")
initial_grant = st.sidebar.number_input("Initial Month/Year 0 Grant (£) [To Kaluza]", value=1000000, step=100000)

# --- ADVANCED OPERATIONAL PARAMETERS (EXPANDABLE) ---
with st.sidebar.expander("4. Operational & Grid Parameters", expanded=False):
    events_per_year = st.number_input("DR Events per Year", min_value=0, value=45, step=1,
                                      help="Average number of Demand Response events called per year based on wind peaks.")
    reliability = st.slider("Response Reliability (%)", min_value=50, max_value=100, value=90,
                            help="The percentage of activated households that successfully shift their load during an event.") / 100
    simultaneous_cap = st.slider("Simultaneous Activation Cap (%)", min_value=10, max_value=100, value=70,
                                 help="Maximum percentage of enrolled households activated at once to prevent grid instability (rebound effect).") / 100
    kwh_per_event = st.number_input("Avg kWh Shifted per Event", value=2.0, step=0.5,
                                    help="The average amount of energy (kWh) shifted by a household during a single DR event.")
    baseline_curtailment_gwh = st.number_input("Baseline Curtailment (Annual GWh)", value=69.0, step=1.0,
                                               help="Total annual curtailed wind energy, used to calculate the % reduction impact.")

with st.sidebar.expander("5. Unit Costs & Revenues", expanded=False):
    st.markdown("**Kaluza Operating Costs & Subscriptions**")
    platform_cost_per_hh = st.number_input("Platform Cost per HH (£/yr)", value=30.0, step=5.0,
                                           help="Annual recurring software, hosting, and support cost per household.")
    fixed_overhead = st.number_input("Fixed Programme Overhead (£/yr)", value=150000, step=10000,
                                     help="Annual fixed costs to administer the programme regardless of household count.")
    monthly_subscription = st.number_input("Monthly Subscription Fee (£/HH)", value=5.0, step=1.0,
                                           help="Monthly fee charged to each household, contributing directly to Kaluza's revenue.")
    reward_per_kwh = st.number_input("Household Reward (£/kWh Shifted)", value=0.03, step=0.01,
                                     help="Direct payment to the household for each kWh successfully shifted during an event.")

    st.markdown("**Market Revenues**")
    wholesale_rate = st.number_input("Wholesale Power Rate (£/MWh)", value=65.0, step=5.0,
                                     help="The baseline wholesale price the Wind Farm gets for selling the un-curtailed electricity.")
    roc_rate = st.number_input("ROC Rate (£/MWh)", value=47.50, step=2.50,
                               help="The value of the Renewables Obligation Certificate earned per MWh of avoided curtailment.")
    dno_rate = st.number_input("DNO Flex Rate (£/MWh)", value=239.0, step=10.0,
                               help="The flexibility service payment rate from the Distribution Network Operator (SSEN).")

with st.sidebar.expander("6. Household Impact Metrics", expanded=False):
    st.markdown("**Consumer Savings Model**")
    avg_annual_bill = st.number_input("Avg Annual Elec Bill (£/HH)", value=700.0, step=100.0,
                                      help="Average household electricity bill before optimization.")
    money_saved_per_kwh = st.number_input("Passive Money Saved (£/kWh)", value=0.15, step=0.01,
                                          help="Estimated passive savings from bill reduction for each kWh of load shifted during a DR event.")

# --- SIMULATION ENGINE (DYNAMIC DURATION & RESOLUTION) ---
periods = list(range(simulation_periods + 1))
results = []
kaluza_cumulative_npv = 0
windfarm_cumulative_npv = 0
hh_previous = 0

for period in periods:
    # 2. Period 0 (Setup Phase - Initial Hardware & Grants)
    if period == 0:
        hh_active = min(initial_hh, max_hh)
        capex = hh_active * total_install_cost * kaluza_capex_split
        kaluza_net_cf = initial_grant - capex
        kaluza_cumulative_npv += kaluza_net_cf
        results.append(
            [period, int(hh_active), 0.0, 0.0, capex, 0.0, 0.0, kaluza_net_cf, kaluza_cumulative_npv, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        hh_previous = hh_active
        continue

    # 1. Calculate Active Households for Period 1 onwards
    if growth_type == "Flat (No Growth)":
        hh_active = initial_hh
    elif growth_type == "Linear (+X per year)":
        hh_active = initial_hh + (growth_rate * (period / annual_divider))
    elif growth_type == "Exponential (+X% per year)":
        hh_active = initial_hh * ((1 + (growth_rate / 100)) ** (period / annual_divider))

    hh_active = min(hh_active, max_hh)
    hh_added = max(0, hh_active - hh_previous)

    # 3. Operational Phase scaled strictly by Active Households & Time Resolution
    kwh_per_single_event = hh_active * simultaneous_cap * kwh_per_event * reliability
    mwh_captured = (kwh_per_single_event * (events_per_year / annual_divider)) / 1000
    total_kwh_shifted = mwh_captured * 1000

    # Kaluza Capex
    capex = hh_added * total_install_cost * kaluza_capex_split

    # Consumer Capex (The portion of the install cost passed to the user)
    consumer_install_cost = total_install_cost * (1.0 - kaluza_capex_split)
    total_hh_install_cost = hh_added * consumer_install_cost

    windfarm_gross_revenue = mwh_captured * (wholesale_rate + roc_rate + dno_rate)
    kaluza_market_share = windfarm_gross_revenue * kaluza_rev_share
    windfarm_net_cf = windfarm_gross_revenue - kaluza_market_share

    subscription_revenue = hh_active * (monthly_subscription * (12 / annual_divider))
    kaluza_revenue = kaluza_market_share + subscription_revenue

    base_opex = (hh_active * (platform_cost_per_hh / annual_divider)) + (fixed_overhead / annual_divider)
    reward_payout = total_kwh_shifted * reward_per_kwh
    opex = base_opex + reward_payout

    kaluza_net_cf = kaluza_revenue - opex - capex

    kaluza_discounted_cf = kaluza_net_cf / ((1 + discount_rate) ** (period / annual_divider))
    kaluza_cumulative_npv += kaluza_discounted_cf
    windfarm_discounted_cf = windfarm_net_cf / ((1 + discount_rate) ** (period / annual_divider))
    windfarm_cumulative_npv += windfarm_discounted_cf

    # Household Level Metrics
    total_hh_elec_cost = hh_active * (avg_annual_bill / annual_divider)
    total_hh_subscription_cost = hh_active * (monthly_subscription * (12 / annual_divider))
    total_hh_savings = (total_kwh_shifted * money_saved_per_kwh) + reward_payout

    # Aggregate Net Cost includes the install cost for the new cohort
    net_hh_cost = total_hh_elec_cost + total_hh_subscription_cost + total_hh_install_cost - total_hh_savings

    if hh_active > 0:
        single_hh_elec_cost = (avg_annual_bill / annual_divider)
        single_hh_subscription = (monthly_subscription * (12 / annual_divider))

        # Single Household hits the install cost ONLY in Period 1
        single_hh_install_cost = consumer_install_cost if period == 1 else 0.0

        single_hh_savings = total_hh_savings / hh_active
        single_net_hh_cost = single_hh_elec_cost + single_hh_subscription + single_hh_install_cost - single_hh_savings
    else:
        single_hh_elec_cost = 0
        single_hh_savings = 0
        single_net_hh_cost = 0

    results.append([
        period, int(hh_active), kwh_per_single_event, mwh_captured, capex, opex, kaluza_revenue, kaluza_net_cf,
        kaluza_cumulative_npv, windfarm_gross_revenue, windfarm_net_cf, windfarm_cumulative_npv,
        total_hh_elec_cost, total_hh_savings, net_hh_cost,
        single_hh_elec_cost, single_hh_savings, single_net_hh_cost
    ])
    hh_previous = hh_active

# --- DATA FORMATTING ---
df = pd.DataFrame(results, columns=[
    period_label, "Households", "kWh per Single Event", "MWh Captured", "Kaluza Capex (£)", "Kaluza Opex (£)",
    "Kaluza Rev (£)", "Kaluza Net CF (£)", "Kaluza NPV (£)",
    "Wind Farm Gross Rev (£)", "Wind Farm Net CF (£)", "Wind Farm NPV (£)",
    "Total HH Elec Cost (£)", "Total HH Savings (£)", "Net HH Elec Cost (£)",
    "Single HH Elec Cost (£)", "Single HH Savings (£)", "Single Net HH Cost (£)"
])

df["kWh Saved Total"] = df["MWh Captured"] * 1000
df["Kaluza Total Cost (£)"] = df["Kaluza Capex (£)"] + df["Kaluza Opex (£)"]

# --- DASHBOARD UI ---
st.subheader(f"Period {simulation_periods} Outcomes ({period_label}ly Forecast)")

col1, col2, col3 = st.columns(3)
col1.metric(f"Kaluza {simulation_periods}-{short_label} NPV", f"£{int(df['Kaluza NPV (£)'].iloc[-1]):,}")
col2.metric(f"Wind Farm {simulation_periods}-{short_label} NPV", f"£{int(df['Wind Farm NPV (£)'].iloc[-1]):,}")
col3.metric("Total Wind Farm Gross Rev", f"£{int(df['Wind Farm Gross Rev (£)'].sum()):,}")

st.write("")

col4, col5, col6 = st.columns(3)
col4.metric("Households Active", f"{int(df['Households'].iloc[-1]):,}")

sim_years = simulation_periods / annual_divider
col5.metric("Curtailment Reduced",
            f"{round((df['MWh Captured'].sum() / (baseline_curtailment_gwh * 1000 * sim_years)) * 100, 2)}%")
col6.metric("Total kWh Saved", f"{int(df['kWh Saved Total'].sum()):,}")

# --- CHART 1: HOUSEHOLD GROWTH ---
st.subheader("Active Households Over Time")
fig_growth = go.Figure()
fig_growth.add_trace(
    go.Scatter(x=df[period_label], y=df["Households"], name="Active Households", mode="lines", fill='tozeroy',
               line=dict(color="royalblue", width=3)))
fig_growth.add_trace(
    go.Scatter(x=[0, simulation_periods], y=[max_hh, max_hh], name="System Capacity (Cap)", mode="lines",
               line=dict(color="red", width=2, dash="dash")))
fig_growth.update_layout(xaxis_title=period_label, yaxis_title="Number of Households", hovermode="x unified",
                         template="plotly_white",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_growth, use_container_width=True)

# --- CHART 2: SINGLE DR EVENT IMPACT ---
st.subheader("Grid Impact: Energy Shifted per Single DR Event")
df_operational = df[df[period_label] > 0]
fig_event = go.Figure()
fig_event.add_trace(
    go.Bar(x=df_operational[period_label], y=df_operational["kWh per Single Event"], name="kWh Shifted (Per Event)",
           marker_color='mediumpurple'))
fig_event.update_layout(xaxis_title=period_label, yaxis_title="Energy (kWh)", hovermode="x unified",
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_event, use_container_width=True)

# --- CHART 3: KALUZA FUNDAMENTALS ---
st.subheader(f"Kaluza {period_label}ly Financial Fundamentals (Revenue vs. Cost)")
fig_fundamentals = go.Figure()
fig_fundamentals.add_trace(
    go.Scatter(x=df_operational[period_label], y=df_operational["Kaluza Total Cost (£)"], name="Kaluza Total Cost",
               mode="lines+markers", line=dict(color="red", width=2)))
fig_fundamentals.add_trace(
    go.Scatter(x=df_operational[period_label], y=df_operational["Kaluza Rev (£)"], name="Kaluza Revenue",
               mode="lines+markers", line=dict(color="green", width=2)))
fig_fundamentals.add_trace(
    go.Bar(x=df_operational[period_label], y=df_operational["Kaluza Net CF (£)"], name="Kaluza Profit (Net CF)",
           marker_color=np.where(df_operational["Kaluza Net CF (£)"] < 0, 'rgba(255, 165, 0, 0.6)',
                                 'rgba(50, 205, 50, 0.6)')))
fig_fundamentals.update_layout(xaxis_title=period_label, yaxis_title="GBP (£)", hovermode="x unified",
                               template="plotly_white",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_fundamentals, use_container_width=True)

# --- CHART 4: WIND FARM REVENUE & PROFIT ---
st.subheader(f"Wind Farm {period_label}ly Revenue vs. Profit")
fig_wf = go.Figure()
fig_wf.add_trace(
    go.Bar(x=df_operational[period_label], y=df_operational["Wind Farm Gross Rev (£)"], name="Gross Revenue",
           marker_color='rgba(169, 169, 169, 0.5)'))
fig_wf.add_trace(
    go.Bar(x=df_operational[period_label], y=df_operational["Wind Farm Net CF (£)"], name="Profit (After Kaluza Share)",
           marker_color='rgba(75, 192, 192, 0.8)'))
fig_wf.update_layout(barmode='group', xaxis_title=period_label, yaxis_title="GBP (£)", hovermode="x unified",
                     template="plotly_white",
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_wf, use_container_width=True)

# --- CHART 5: CUMULATIVE NPV & SHARED CASHFLOWS ---
st.subheader(f"{period_label}ly Cumulative NPV & Partnership Cashflows")
fig_npv = go.Figure()
fig_npv.add_trace(
    go.Bar(x=df_operational[period_label], y=df_operational["Kaluza Net CF (£)"], name="Kaluza Net Cashflow",
           marker_color='rgba(255, 99, 132, 0.7)'))
fig_npv.add_trace(
    go.Bar(x=df_operational[period_label], y=df_operational["Wind Farm Net CF (£)"], name="Wind Farm Net Profit",
           marker_color='rgba(75, 192, 192, 0.7)'))
fig_npv.add_trace(go.Scatter(x=df[period_label], y=df["Kaluza NPV (£)"], name="Kaluza Cumulative NPV", mode="lines",
                             line=dict(color="red", width=3)))
fig_npv.add_trace(
    go.Scatter(x=df[period_label], y=df["Wind Farm NPV (£)"], name="Wind Farm Cumulative NPV", mode="lines",
               line=dict(color="teal", width=3)))
fig_npv.update_layout(barmode='group', xaxis_title=period_label, yaxis_title="GBP (£)", hovermode="x unified",
                      template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_npv, use_container_width=True)

# --- CHART 6: HOUSEHOLD COST & SAVINGS (DYNAMIC COLORS) ---
st.subheader(f"{period_label}ly Household Electricity Costs vs. Savings")
view_mode = st.radio(
    "Select View Mode:",
    ["Total Active Households (Aggregate)", "Single Household (Per-User)"],
    horizontal=True
)

fig_hh = go.Figure()
if view_mode == "Total Active Households (Aggregate)":
    y_net_cost = df_operational["Net HH Elec Cost (£)"]
    y_savings = df_operational["Total HH Savings (£)"]
    y_original = df_operational["Total HH Elec Cost (£)"]
else:
    y_net_cost = df_operational["Single Net HH Cost (£)"]
    y_savings = df_operational["Single HH Savings (£)"]
    y_original = df_operational["Single HH Elec Cost (£)"]

# Dynamic coloring: Red if Net Cost > Original Bill, Blue if they are saving money
net_bar_colors = np.where(y_net_cost > y_original, 'rgba(255, 99, 132, 0.8)', 'rgba(54, 162, 235, 0.8)')

fig_hh.add_trace(go.Bar(
    x=df_operational[period_label],
    y=y_net_cost,
    name=f"Net {period_label}ly Cost Paid",
    marker_color=net_bar_colors
))

# Savings bar is green
fig_hh.add_trace(go.Bar(
    x=df_operational[period_label],
    y=y_savings,
    name=f"{period_label}ly Money Saved",
    marker_color='rgba(50, 205, 50, 0.6)'
))

fig_hh.add_trace(go.Scatter(
    x=df_operational[period_label],
    y=y_original,
    name=f"Original {period_label}ly Cost (Unoptimized)",
    mode="lines",
    line=dict(color="orange", width=2, dash="dash")
))

fig_hh.update_layout(
    barmode='stack',
    xaxis_title=period_label,
    yaxis_title="GBP (£)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_hh, use_container_width=True)

# --- DATA TABLE ---
st.subheader(f"Raw {period_label}ly Data Table")
st.dataframe(df.style.format({
    "kWh per Single Event": "{:,.0f}",
    "kWh Saved Total": "{:,.0f}",
    "MWh Captured": "{:.1f}",
    "Kaluza Capex (£)": "£{:,.0f}",
    "Kaluza Opex (£)": "£{:,.0f}",
    "Kaluza Total Cost (£)": "£{:,.0f}",
    "Kaluza Rev (£)": "£{:,.0f}",
    "Kaluza Net CF (£)": "£{:,.0f}",
    "Kaluza NPV (£)": "£{:,.0f}",
    "Wind Farm Gross Rev (£)": "£{:,.0f}",
    "Wind Farm Net CF (£)": "£{:,.0f}",
    "Wind Farm NPV (£)": "£{:,.0f}",
    "Total HH Elec Cost (£)": "£{:,.0f}",
    "Total HH Savings (£)": "£{:,.0f}",
    "Net HH Elec Cost (£)": "£{:,.0f}",
    "Single HH Elec Cost (£)": "£{:,.1f}",
    "Single HH Savings (£)": "£{:,.2f}",
    "Single Net HH Cost (£)": "£{:,.1f}"
}))