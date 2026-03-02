import os
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Orkney Energy Dashboard 2017", layout="wide")
st.title("Orkney Energy Dashboard 2017")

# ── Constants ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
TURBINE_CUTOFF = datetime.date(2017, 12, 27)

MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

SEASONS = {
    "Spring (Mar-May)": [3, 4, 5],
    "Summer (Jun-Aug)": [6, 7, 8],
    "Autumn (Sep-Nov)": [9, 10, 11],
    "Winter (Dec-Feb)": [12, 1, 2],
}

# Orkney monthly avg temperatures °C (midpoint of max/min, Loch of Hundland station)
ORKNEY_MONTHLY_TEMP_C = {
    1: 4.20, 2: 4.15, 3: 5.14, 4: 6.98,
    5: 9.12, 6: 11.29, 7: 13.34, 8: 13.30,
    9: 11.67, 10: 8.95, 11: 6.39, 12: 4.48,
}
# Approximate day-of-year for mid-month
_MONTH_MIDDAYS = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]


# ── Gap-filled turbine data generator ────────────────────────────────────────
@st.cache_data
def load_or_generate_fixed_turbine():
    """Load turbine_telemetry_fixed.csv if it exists, else generate it.

    Strategy: Apr 4–23 2017 is a complete NaN gap (960 rows with missing Wind_ms,
    Power_kw, Setpoint_kw). We fill Wind_ms only, using the same calendar block
    from April 2016 in the original 1-min resolution CSV, resampled to 30-min means.
    Power_kw/Setpoint_kw remain NaN — the dashboard derives actual power from the LUT
    using Wind_ms, so filling Wind_ms is sufficient.
    """
    fixed_path = os.path.join(BASE, "turbine_telemetry_fixed.csv")
    if os.path.exists(fixed_path):
        return pd.read_csv(fixed_path, parse_dates=["Timestamp"])

    # ── Load original 1-min resolution CSV ───────────────────────────────────
    orig_path = os.path.join(BASE, "..", "Supplied Data", "Turbine_telemetry.csv")
    orig = pd.read_csv(
        orig_path,
        usecols=[0, 3],           # Timestamp, Wind_ms columns
        names=["Timestamp", "Wind_ms"],
        header=0,
        dayfirst=True,
        parse_dates=["Timestamp"],
    )
    orig = orig.dropna(subset=["Timestamp"])
    orig["Timestamp"] = pd.to_datetime(orig["Timestamp"], dayfirst=True, errors="coerce")
    orig = orig.dropna(subset=["Timestamp"])

    # ── Extract April 4–23, 2016 and resample to 30-min means ────────────────
    apr2016 = orig[
        (orig["Timestamp"].dt.year == 2016) &
        (orig["Timestamp"].dt.month == 4) &
        (orig["Timestamp"].dt.day >= 4) &
        (orig["Timestamp"].dt.day <= 23)
    ].copy()
    apr2016 = apr2016.set_index("Timestamp").sort_index()
    wind_30min = apr2016["Wind_ms"].resample("30min").mean()

    # Build a mapping: (month, day, hour, minute) → wind speed
    wind_lookup = {
        (ts.month, ts.day, ts.hour, ts.minute): val
        for ts, val in wind_30min.items()
        if pd.notna(val)
    }

    # ── Load the 2017 base CSV ────────────────────────────────────────────────
    base_path = os.path.join(BASE, "Turbine_telemetry_2017.csv")
    df = pd.read_csv(base_path, parse_dates=["Timestamp"])

    # ── Fill Wind_ms for the gap rows using 2016 values ──────────────────────
    gap_mask = (
        (df["Timestamp"].dt.month == 4) &
        (df["Timestamp"].dt.day >= 4) &
        (df["Timestamp"].dt.day <= 23) &
        df["Wind_ms"].isna()
    )
    # Also fill Apr 3 15:30–23:30 and Apr 24 00:00–02:30 partial gaps
    gap_mask_ext = (
        (
            (df["Timestamp"].dt.month == 4) &
            (df["Timestamp"].dt.day == 3) &
            (df["Timestamp"].dt.hour >= 15) &
            df["Wind_ms"].isna()
        ) | gap_mask | (
            (df["Timestamp"].dt.month == 4) &
            (df["Timestamp"].dt.day == 24) &
            (df["Timestamp"].dt.hour < 3) &
            df["Wind_ms"].isna()
        )
    )

    def fill_wind(ts):
        key = (ts.month, ts.day, ts.hour, ts.minute)
        return wind_lookup.get(key, np.nan)

    df.loc[gap_mask_ext, "Wind_ms"] = df.loc[gap_mask_ext, "Timestamp"].apply(fill_wind)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    df.to_csv(fixed_path, index=False)
    return df


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    demand = pd.read_csv(
        os.path.join(BASE, "Residential_demand_2017.csv"),
        parse_dates=["Timestamp"],
    )
    turbine = pd.read_csv(
        os.path.join(BASE, "Turbine_telemetry_2017.csv"),
        parse_dates=["Timestamp"],
    )
    turbine_fixed = load_or_generate_fixed_turbine()

    # ── Power curve: clean erroneous zero readings ────────────────────────────
    # Zeros where wind > 3 m/s are instrument/reporting errors, not real cut-outs
    CUT_IN_MS = 3.0
    clean = turbine.dropna(subset=["Power_kw", "Wind_ms"])
    clean = clean[~((clean["Power_kw"] == 0) & (clean["Wind_ms"] > CUT_IN_MS))]

    # ── Temperature Fourier fit: day-of-year → °C ────────────────────────────
    # 2-harmonic Fourier series: naturally periodic over 365 days,
    # so December wraps correctly back to January (unlike a polynomial).
    temps  = np.array([ORKNEY_MONTHLY_TEMP_C[m] for m in range(1, 13)])
    doys   = np.array(_MONTH_MIDDAYS, dtype=float)
    _omega = 2.0 * np.pi / 365.0
    _Xt    = np.column_stack([
        np.ones(12),
        np.cos(_omega * doys), np.sin(_omega * doys),
        np.cos(2.0 * _omega * doys), np.sin(2.0 * _omega * doys),
    ])
    temp_coeffs, _, _, _ = np.linalg.lstsq(_Xt, temps, rcond=None)
    # temp_coeffs: 5-element array [a0, a1_cos, b1_sin, a2_cos, b2_sin]

    return demand, turbine, turbine_fixed, clean, temp_coeffs


demand_df, turbine_df, turbine_fixed_df, turbine_clean_df, temp_coeffs = load_data()


# ── Temperature helper (Fourier) ──────────────────────────────────────────────
def temp_from_doy(doy):
    """Evaluate Fourier temperature model. doy may be scalar or array."""
    doy = np.asarray(doy, dtype=float)
    scalar = doy.ndim == 0
    doy = np.atleast_1d(doy)
    _omega = 2.0 * np.pi / 365.0
    X = np.column_stack([
        np.ones(len(doy)),
        np.cos(_omega * doy), np.sin(_omega * doy),
        np.cos(2.0 * _omega * doy), np.sin(2.0 * _omega * doy),
    ])
    result = X @ temp_coeffs
    return float(result[0]) if scalar else result


# ── Piecewise power curve helpers ─────────────────────────────────────────────
@st.cache_data
def build_power_lut(cut_in, cut_out, rated_kw, rated_wind_speed_override=None,
                    bin_width=0.2, use_fixed=False):
    """Build a wind-speed → power look-up table (LUT) from median binned telemetry.

    Approach mirrors Data Brief 1:
    1. Filter to unconstrained records: Setpoint_kw == 900, drop zeros at above cut-in
    2. Bin wind speeds at `bin_width` m/s resolution
    3. Compute median power per bin (robust to outliers)
    4. Enforce physical constraints: 0 below cut-in, rated_kw above rated wind speed,
       0 above cut-out
    5. Return (lut_wind, lut_power, rated_wind_speed, clean_df) for interpolation

    If `rated_wind_speed_override` is provided, it replaces the auto-detected rated speed.
    use_fixed selects the gap-filled dataset for the LUT fit.
    """
    src = turbine_fixed_df if use_fixed else turbine_df
    clean = src.dropna(subset=["Power_kw", "Wind_ms", "Setpoint_kw"])
    # Only use fully unconstrained data (setpoint at max)
    clean = clean[clean["Setpoint_kw"] == 900]
    # Remove erroneous zero-power readings above cut-in
    clean = clean[~((clean["Power_kw"] == 0) & (clean["Wind_ms"] > cut_in))]
    # Remove above cut-out (storm shutdowns)
    clean = clean[clean["Wind_ms"] <= cut_out]

    # Bin at bin_width resolution and compute median (keep only populated bins)
    clean_copy = clean.copy()
    clean_copy["wind_bin"] = (clean_copy["Wind_ms"] / bin_width).round() * bin_width
    lut = clean_copy.groupby("wind_bin")["Power_kw"].median()

    # Cap at rated_kw
    lut = lut.clip(upper=rated_kw)

    # Identify rated wind speed: first bin where median >= 95% of rated
    above = lut[lut >= 0.95 * rated_kw]
    auto_rated_wind_speed = float(above.index.min()) if not above.empty else float(clean["Wind_ms"].quantile(0.90))

    # Use override if provided, otherwise use auto-detected value
    rated_wind_speed = float(rated_wind_speed_override) if rated_wind_speed_override is not None else auto_rated_wind_speed

    # Enforce flat rated plateau from rated_wind_speed onwards
    lut[lut.index >= rated_wind_speed] = rated_kw

    # Add anchor at (0, 0) so the interpolation starts from the origin
    lut[0.0] = 0.0
    lut = lut[lut.index <= cut_out].sort_index()
    # Zero out all bins at or below cut-in (no power below cut-in)
    lut[lut.index <= cut_in] = 0.0

    lut_wind = lut.index.values.astype(float)
    lut_power = lut.values.astype(float)
    return lut_wind, lut_power, rated_wind_speed, auto_rated_wind_speed, clean, lut


def eval_power_curve(wind_ms, lut_wind, lut_power, cut_in, rated_wind_speed, cut_out, rated_kw):
    """Evaluate power curve via linear interpolation of the LUT.

    Segments:
      wind < cut_in       → 0
      cut_in ≤ wind ≤ cut_out → linear interp between LUT bin centres
      wind > cut_out      → 0
    Works on scalars and arbitrary-shape ndarrays.
    """
    wind = np.asarray(wind_ms, dtype=float)
    scalar = wind.ndim == 0
    wind = np.atleast_1d(wind)
    interp = np.interp(wind, lut_wind, lut_power, left=0.0, right=0.0)
    power = np.where(
        wind < cut_in, 0.0,
        np.where(wind <= cut_out, interp, 0.0)
    )
    return float(power.flat[0]) if scalar else power


# ── Pre-compute dataset-level facts ──────────────────────────────────────────
_n_hh_sample = int(demand_df["N_households"].median())
_turbine_null_intervals = int(turbine_df["Power_kw"].isna().sum())
_turbine_null_days = round(_turbine_null_intervals * 0.5 / 24, 1)
_zeros_removed = int(
    ((turbine_df["Power_kw"] == 0) & (turbine_df["Wind_ms"] > 3.0)).sum()
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Inputs")
    num_households = st.number_input(
        "Number of households", min_value=1, max_value=100000, value=44000, step=500,
        help="Total number of residential households in the modelled area. "
             "Scales the per-household mean demand (`Demand_mean_kw`) to a fleet-wide total.",
    )
    num_turbines = st.number_input(
        "Number of turbines", min_value=1, max_value=639, value=639, step=1,
        help="Number of wind turbines in the modelled wind farm. "
             "Total turbine power = single-turbine LUT power × number of turbines.",
    )
    export_enabled = st.toggle(
        "Export",
        value=False,
        help="When ON, surplus turbine power above demand is exported to the grid "
             "(capped at 40,000 kW). Exported energy is subtracted before curtailment is calculated.",
    )

    st.divider()

    st.markdown("""
**About this dashboard**

Use the inputs above to set the number of residential households and wind turbines.
Select a tab to explore energy supply and demand across different time periods.

- **Daily** — 30-minute resolution for a chosen day
- **Monthly** — daily totals for a chosen month
- **Seasonal** — daily totals across a chosen season
- **Yearly** — daily totals across the full year
- **Analysis Settings** — model parameters and diagnostic plots
""")

    st.divider()

    st.markdown("""
**Useful dates**

- **02/08** - Max supply
- **07/11** - Lowest supply
- **12/07** - Storm shutdown (max wind > cut-out, no generation)

""")

    st.divider()
    
    st.info(
        f"**Demand baseline:** `Demand_mean_kw` is the mean power per household, "
        f"sampled from ~{_n_hh_sample:,} metered households. "
        "The 'Number of households' input scales this to your chosen fleet size."
    )
    st.warning(
        f"**Missing turbine data:** {_turbine_null_intervals} half-hour intervals "
        f"(~{_turbine_null_days} days) have no turbine readings and are excluded from all calculations. "
        f"{_zeros_removed} erroneous zero-power readings (wind > 3 m/s) have been removed from the power curve fit."
    )



   


# ── Physics helpers ───────────────────────────────────────────────────────────
def air_density_from_doy(doy):
    """Return air density (kg/m³) for day-of-year scalar or array."""
    T = temp_from_doy(doy)
    return 1.225 * (273.15 / (273.15 + T))


def potential_power_kw_vec(wind_ms, doy, rotor_diameter_m, n_turbines, cp=0.45):
    """Vectorised Cp × ½ρAv³ in kW for a fleet of n_turbines.

    wind_ms, doy : 1-D arrays of equal length (one value per 30-min interval)
    cp           : power coefficient (default 0.45; Betz limit = 0.593)
    """
    rho = air_density_from_doy(doy)
    A = np.pi * (rotor_diameter_m / 2) ** 2
    return cp * 0.5 * rho * A * (wind_ms ** 3) * n_turbines / 1000


# ── Helper: compute per-interval merged df ────────────────────────────────────
def build_merged(demand_slice, turbine_slice,
                 rotor_diameter_m, rated_power_kw, curtail_on_potential,
                 lut_wind, lut_power, rated_wind_speed_fit, cut_in_speed, cut_out_speed,
                 cp_factor=0.45, availability_factor=1.0):
    """Return interval-level DataFrame with actual_kw, potential_kw, curtailed_kw, etc.

    Turbine Power ≤ Potential always holds: both actual_kw and potential_kw are zeroed
    above cut-out speed (storm shutdown). actual_kw = 0 via eval_power_curve (returns 0
    above cut-out); potential_kw = 0 via explicit storm_mask zeroing below.
    """
    d = demand_slice[["Timestamp", "Demand_mean_kw"]].copy()
    d["demand_kw"] = d["Demand_mean_kw"] * num_households

    t = turbine_slice[["Timestamp", "Power_kw", "Wind_ms"]].copy()

    # ── Per-turbine wind speed perturbation ───────────────────────────────────
    rng = np.random.default_rng(seed=42)
    wind = t["Wind_ms"].fillna(0.0).values  # (T,)

    if wind_spread_sigma == 0.0:
        wind_perturbed = wind[:, None] * np.ones((1, int(num_turbines)))
    else:
        wind_perturbed = rng.normal(
            loc=wind[:, None],
            scale=wind_spread_sigma,
            size=(len(wind), int(num_turbines)),
        ).clip(0)  # (T, N)

    # ── Turbine power via LUT interpolation ──────────────────────────────────
    # eval_power_curve returns 0 above cut-out, so turbine_kw = 0 during storms.
    power_matrix = eval_power_curve(
        wind_perturbed, lut_wind, lut_power, cut_in_speed, rated_wind_speed_fit,
        cut_out_speed, rated_power_kw,
    )  # (T, N)
    t["actual_kw"] = power_matrix.sum(axis=1) * availability_factor

    # ── Potential power via Cp × ½ρAv³ ───────────────────────────────────────
    doy = t["Timestamp"].dt.day_of_year.values.astype(float)
    t["potential_kw"] = potential_power_kw_vec(wind, doy, rotor_diameter_m, num_turbines, cp=cp_factor)
    # Storm shutdown: turbines cannot operate above cut-out speed.
    # Store the would-be potential as storm_shutdown_kw, then zero out both
    # potential_kw and actual_kw so that all three are 0 during storms.
    # actual_kw is also zeroed explicitly because wind perturbation (wind_spread_sigma > 0)
    # can push some per-turbine winds below cut-out even when the mean exceeds it,
    # causing eval_power_curve to return non-zero for those turbines.
    storm_mask = t["Wind_ms"] > cut_out_speed
    t["storm_shutdown_kw"] = t["potential_kw"].where(storm_mask, 0.0)
    t.loc[storm_mask, "potential_kw"] = 0.0
    t.loc[storm_mask, "actual_kw"]    = 0.0

    m = pd.merge(
        d[["Timestamp", "demand_kw"]],
        t[["Timestamp", "actual_kw", "potential_kw", "storm_shutdown_kw", "Wind_ms"]],
        on="Timestamp",
    )

    # ── Curtailment ───────────────────────────────────────────────────────────
    # Surplus = generation above demand (before export absorbs any excess)
    actual_surplus = (m["actual_kw"] - m["demand_kw"]).clip(lower=0)

    if export_enabled:
        m["export_kw"] = actual_surplus.clip(upper=40_000)
    else:
        m["export_kw"] = 0.0

    if curtail_on_potential:
        # Theoretical basis: curtailment = potential not served (independent of export)
        m["curtailed_kw"] = (m["potential_kw"] - m["demand_kw"]).clip(lower=0)
    else:
        # Actual basis: curtailment = actual surplus after export is absorbed
        m["curtailed_kw"] = (actual_surplus - m["export_kw"]).clip(lower=0)

    m["unmet_kw"] = (m["demand_kw"] - m["actual_kw"]).clip(lower=0)
    m["curtailed_interval"] = (m["curtailed_kw"] > curtail_threshold_kw).astype(int)
    m["storm_shutdown_interval"] = (m["storm_shutdown_kw"] > 0).astype(int)
    return m


# ── Helper: aggregate to daily totals (kWh) ──────────────────────────────────
def daily_totals(demand_slice, turbine_slice,
                 rotor_diameter_m, rated_power_kw, curtail_on_potential,
                 lut_wind, lut_power, rated_wind_speed_fit, cut_in_speed, cut_out_speed,
                 cp_factor=0.45, availability_factor=1.0):
    m = build_merged(demand_slice, turbine_slice,
                     rotor_diameter_m, rated_power_kw, curtail_on_potential,
                     lut_wind, lut_power, rated_wind_speed_fit, cut_in_speed, cut_out_speed,
                     cp_factor=cp_factor, availability_factor=availability_factor)
    m["date"] = m["Timestamp"].dt.normalize()
    energy = m.groupby("date")[
        ["demand_kw", "actual_kw", "potential_kw", "curtailed_kw",
         "export_kw", "unmet_kw", "storm_shutdown_kw"]
    ].sum() * 0.5
    energy.columns = [
        "demand_kwh", "actual_kwh", "potential_kwh",
        "curtailed_kwh", "export_kwh", "unmet_kwh", "storm_shutdown_kwh",
    ]
    intervals = m.groupby("date")[["curtailed_interval", "storm_shutdown_interval"]].sum()
    totals = energy.join(intervals)
    totals["curtailed_hours"] = totals["curtailed_interval"] * 0.5
    totals["storm_shutdown_hours"] = totals["storm_shutdown_interval"] * 0.5
    totals.drop(columns=["curtailed_interval", "storm_shutdown_interval"], inplace=True)
    return totals


# ── Helper: build a 4-line Plotly figure (aggregate tabs) ────────────────────
def make_figure(x, demand, potential, actual, curtailed, title, y_label,
                storm_shutdown=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=demand, name="Demand", mode="lines",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="%{x}<br>Demand: %{y:,.0f} " + y_label + "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=potential, name="Potential Power", mode="lines",
        line=dict(color="#aec7e8", width=1.5, dash="dash"),
        hovertemplate="%{x}<br>Potential: %{y:,.0f} " + y_label + "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=actual, name="Turbine Power", mode="lines",
        line=dict(color="#ff7f0e", width=2),
        hovertemplate="%{x}<br>Turbine: %{y:,.0f} " + y_label + "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=curtailed, name="Curtailed", mode="lines",
        line=dict(color="#189e07", width=2, dash="dashdot"),
        hovertemplate="%{x}<br>Curtailed: %{y:,.0f} " + y_label + "<extra></extra>",
    ))
    if storm_shutdown is not None:
        fig.add_trace(go.Scatter(
            x=x, y=storm_shutdown, name="Storm Shutdown", mode="lines",
            line=dict(color="#9467bd", width=1.5, dash="dot"),
            hovertemplate="%{x}<br>Storm Shutdown: %{y:,.0f} " + y_label + "<extra></extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Time" if y_label == "kW" else "Date",
        yaxis_title=f"Power ({y_label})" if y_label == "kW" else f"Energy ({y_label})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
    )
    return fig


# ── Helper: summary metrics ───────────────────────────────────────────────────
def show_metrics(total_d, total_actual, total_potential, total_c,
                 total_exp, total_unmet, total_c_hrs, prefix="Total",
                 total_storm_hrs=0):
    curtail_pct  = 100 * total_c / total_potential if total_potential > 0 else 0.0
    unmet_pct    = 100 * total_unmet / total_d if total_d > 0 else 0.0
    self_suff    = 100 * (total_d - total_unmet) / total_d if total_d > 0 else 0.0
    cap_fac      = 100 * total_actual / total_potential if total_potential > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{prefix} Demand",          f"{total_d:,.0f} kWh",
                help="Total electrical energy consumed by all households over the period. "
                     "Derived from scaled per-household mean demand × number of households.")
    col2.metric(f"{prefix} Potential Power",  f"{total_potential:,.0f} kWh",
                help="Maximum wind energy theoretically extractable: Cₚ × ½ρAv³ × turbines. "
                     "Represents what the wind resource could deliver with no grid or mechanical constraints. "
                     "Zeroed during storm shutdowns (wind > cut-out speed).")
    col3.metric(f"{prefix} Turbine Power",    f"{total_actual:,.0f} kWh",
                delta=f"{cap_fac:.1f}% of potential",
                help="Energy actually delivered by the turbines, derived from the LUT power curve "
                     "fitted to unconstrained telemetry, then scaled by availability factor. "
                     "The delta shows this as a percentage of Potential Power (capacity factor proxy).")

    col4, col5, col6 = st.columns(3)
    col4.metric(f"{prefix} Curtailed",    f"{total_c:,.0f} kWh",
                delta=f"{curtail_pct:.1f}% of potential", delta_color="inverse",
                help="Energy that could not be used: turbine surplus above demand after export. "
                     "Curtailment = max(0, actual − demand − export). "
                     "If 'theoretical basis' is enabled, uses potential power instead of actual.")
    col5.metric(f"{prefix} Exported",     f"{total_exp:,.0f} kWh",
                help="Energy exported to the wider grid when turbine output exceeds local demand. "
                     "Capped at 40,000 kW per interval. Only active when the Export toggle is ON.")
    col6.metric(f"{prefix} Unmet Demand", f"{total_unmet:,.0f} kWh",
                delta=f"{unmet_pct:.1f}% of demand", delta_color="inverse",
                help="Energy demand that could not be met by turbine output. "
                     "Unmet = max(0, demand − turbine power). "
                     "Occurs when wind is low, turbines are unavailable, or during storm shutdown.")

    col7, col8, col9 = st.columns(3)
    col7.metric("Self-Sufficiency",    f"{self_suff:.1f}%",
                help="Percentage of total demand met by local turbine generation. "
                     "Self-sufficiency = (demand − unmet) / demand × 100. "
                     "100% means all demand was covered by wind at every interval.")
    col8.metric("Curtailment Hours",   f"{total_c_hrs:,.1f} h",
                help=f"Number of hours where curtailed power exceeded the {curtail_threshold_kw:,} kW threshold. "
                     f"Each qualifying 30-min interval counts as 0.5 h. "
                     f"Adjust the threshold in Analysis Settings.")
    col9.metric("Storm Shutdown Hours", f"{total_storm_hrs:,.1f} h",
                help="Number of hours where wind speed exceeded the cut-out speed, "
                     "forcing all turbines to shut down for safety. "
                     "Distinct from curtailment — turbines are physically off, not grid-constrained.")


# ── Setting defaults (overridden inside tab_settings on every run) ────────────
# These must be defined before the tab blocks so that Daily/Monthly/Seasonal/Yearly
# can reference them. Streamlit always executes all tab blocks on every run.
wind_spread_sigma   = 1.0
curtail_threshold_kw = 3925
rotor_diameter_m    = 44.0
rated_power_kw      = 900
cp_factor           = 0.45
cut_in_speed        = 3.0
cut_out_speed       = 25.0
curtail_on_potential = False
use_fixed_data      = False
availability_factor  = 0.85

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_daily, tab_monthly, tab_seasonal, tab_yearly, tab_settings = st.tabs(
    ["Daily", "Monthly", "Seasonal", "Yearly", "Analysis Settings"]
)

# ── Analysis Settings ─────────────────────────────────────────────────────────
with tab_settings:
    st.header("Model Parameters")

    col_a, col_b = st.columns(2)
    with col_a:
        rotor_diameter_m = st.number_input(
            "Rotor diameter (m)", min_value=1.0, max_value=200.0, value=44.0, step=0.5,
            help="Enercon E-44 = 44 m. Swept area A = π(D/2)²",
        )
        rated_power_kw = st.number_input(
            "Rated power cap per turbine (kW)",
            min_value=100, max_value=5000, value=900, step=50,
            help="Maximum nameplate power output of a single turbine. "
                 "The Enercon E-44 used here is rated at 900 kW. "
                 "The LUT power curve is capped at this value above rated wind speed.",
        )
        cp_factor = st.number_input(
            "Power coefficient C\u209a",
            min_value=0.10, max_value=0.593, value=0.45, step=0.01,
            help="Fraction of wind kinetic energy the turbine can extract. "
                 "Betz limit = 0.593 (theoretical max). Typical modern turbines: 0.35\u20130.48. "
                 "Applied to P = \u00bdρAv\u00b3 to give realistic 'Potential Power'.",
        )
        availability_factor = st.number_input(
            "Turbine availability factor",
            min_value=0.01, max_value=1.00, value=0.85, step=0.01,
            help="Fraction of time turbines are operational (accounts for planned and "
                 "unplanned downtime). 1.0 = always available. Typical onshore wind: 0.85\u20130.97. "
                 "Applied as a scalar multiplier to turbine power output.",
        )
    with col_b:
        cut_in_speed = st.number_input(
            "Cut-in wind speed (m/s)", min_value=0.0, max_value=10.0, value=3.0, step=0.5,
            help="Zero-power readings above this speed are treated as erroneous and excluded from the power curve fit.",
        )
        cut_out_speed = st.number_input(
            "Cut-out wind speed (m/s)", min_value=10.0, max_value=50.0, value=25.0, step=0.5,
            help="Turbine shuts down above this speed. Rated power is held flat from rated wind speed to cut-out.",
        )

    col_c, col_d = st.columns(2)
    with col_c:
        wind_spread_sigma = st.number_input(
            "Wind speed spread \u03c3 (m/s)",
            min_value=0.0, max_value=5.0, value=1.0, step=0.1,
            help="Standard deviation of the Gaussian wind speed variation applied independently "
                 "to each turbine. Simulates spatial wind variability across the farm. "
                 "0 = all turbines see identical wind. Higher values spread output across the power curve, "
                 "smoothing the aggregate generation profile.",
        )
    with col_d:
        curtail_threshold_kw = st.number_input(
            "Curtailment threshold (kW)",
            min_value=0, max_value=500_000, value=3925, step=100,
            help="Minimum curtailed power for a 30-min interval to be counted towards "
                 "'Curtailment Hours'. Set to 0 to count any non-zero curtailment. "
                 "The default (3,925 kW) is roughly one full turbine at rated output — "
                 "filtering out minor grid imbalances.",
        )

    curtail_on_potential = st.toggle(
        "Use theoretical potential power as curtailment basis",
        value=False,
        help=(
            "**OFF (default):** Curtailment = actual surplus after demand and export "
            "\u2014 i.e. real curtailed generation (actual_kw \u2212 demand_kw \u2212 export_kw).\n\n"
            "**ON:** Curtailment = theoretical wind energy not served "
            "(potential_kw \u2212 demand_kw), regardless of what turbines actually output."
        ),
    )

    use_fixed_data = st.toggle(
        "Use gap-filled turbine data (Apr 4\u201323 synthetic wind)",
        value=True,
        help=(
            "**OFF:** Original Turbine_telemetry_2017.csv — Apr 4\u201323 is a data gap (NaN).\n\n"
            "**ON (default):** turbine_telemetry_fixed.csv — Wind_ms for Apr 4\u201323 filled from April 2016 data "
            "(same calendar block, resampled from 1-min to 30-min). Power_kw remains NaN; "
            "turbine output is derived from wind via the LUT as usual."
        ),
    )

    # Build LUT once without override to get the auto-detected rated wind speed
    _, _, _, auto_rated_ws, _, _ = build_power_lut(
        cut_in_speed, cut_out_speed, rated_power_kw, use_fixed=use_fixed_data
    )

    rated_wind_speed_input = st.number_input(
        "Rated wind speed (m/s)",
        min_value=float(cut_in_speed) + 0.5,
        max_value=float(cut_out_speed) - 0.5,
        value=round(auto_rated_ws * 2) / 2,  # round to nearest 0.5 for cleaner default
        step=0.5,
        help=f"Wind speed at which the turbine reaches rated power. "
             f"Auto-detected from data: {auto_rated_ws:.1f} m/s. Adjust if the plateau starts too early or too late.",
    )

    # Rebuild LUT with the user-chosen rated wind speed
    lut_wind, lut_power, rated_wind_speed_fit, _, turbine_clean, lut_series = build_power_lut(
        cut_in_speed, cut_out_speed, rated_power_kw,
        rated_wind_speed_override=rated_wind_speed_input,
        use_fixed=use_fixed_data,
    )

    st.divider()
    st.subheader("Diagnostic Plots")

    diag1, diag2, diag3 = st.columns(3)

    # ── Plot 1: Power curve ───────────────────────────────────────────────────
    with diag1:
        st.markdown("**Power vs Wind Speed**")
        w_range = np.linspace(0, cut_out_speed + 1, 400)
        p_fit = eval_power_curve(
            w_range, lut_wind, lut_power, cut_in_speed, rated_wind_speed_fit, cut_out_speed, rated_power_kw
        )
        fig_pc = go.Figure()
        fig_pc.add_trace(go.Scatter(
            x=turbine_clean["Wind_ms"], y=turbine_clean["Power_kw"],
            mode="markers", name="Cleaned data",
            marker=dict(color="#aec7e8", size=3, opacity=0.4),
        ))
        fig_pc.add_trace(go.Scatter(
            x=w_range, y=p_fit, mode="lines", name="LUT curve",
            line=dict(color="#ff7f0e", width=2.5),
        ))
        fig_pc.add_trace(go.Scatter(
            x=lut_series.index, y=lut_series.values,
            mode="markers", name="Bin medians",
            marker=dict(color="#d62728", size=5, symbol="circle"),
        ))
        fig_pc.add_vline(x=cut_in_speed, line_dash="dot", line_color="gray",
                         annotation_text=f"Cut-in {cut_in_speed} m/s",
                         annotation_position="top right")
        fig_pc.add_vline(x=rated_wind_speed_fit, line_dash="dot", line_color="green",
                         annotation_text=f"Rated {rated_wind_speed_fit:.1f} m/s",
                         annotation_position="top right")
        fig_pc.add_vline(x=cut_out_speed, line_dash="dot", line_color="red",
                         annotation_text=f"Cut-out {cut_out_speed} m/s",
                         annotation_position="top left")
        fig_pc.update_layout(
            xaxis_title="Wind Speed (m/s)", yaxis_title="Power (kW)",
            margin=dict(t=20, b=40), height=320,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_pc, use_container_width=True)
        auto_note = f" (auto-detected: {auto_rated_ws:.1f} m/s)" if abs(rated_wind_speed_input - auto_rated_ws) > 0.1 else " (auto-detected)"
        st.caption(f"Rated wind speed set to {rated_wind_speed_fit:.1f} m/s{auto_note}. "
                   f"Flat plateau enforced from this speed to cut-out.")

    # ── Plot 2: Temperature vs month ──────────────────────────────────────────
    with diag2:
        st.markdown("**Temperature vs Day of Year**")
        doy_range = np.linspace(1, 365, 365)
        temp_fit = temp_from_doy(doy_range)
        monthly_temps = [ORKNEY_MONTHLY_TEMP_C[m] for m in range(1, 13)]
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=_MONTH_MIDDAYS, y=monthly_temps,
            mode="markers", name="Monthly avg",
            marker=dict(color="#1f77b4", size=8),
        ))
        fig_temp.add_trace(go.Scatter(
            x=doy_range, y=temp_fit, mode="lines", name="Fourier fit",
            line=dict(color="#d62728", width=2),
        ))
        fig_temp.update_layout(
            xaxis_title="Day of Year", yaxis_title="Temperature (°C)",
            margin=dict(t=20, b=40), height=320,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    # ── Plot 3: Air density vs month ──────────────────────────────────────────
    with diag3:
        st.markdown("**Air Density vs Day of Year**")
        rho_fit = air_density_from_doy(doy_range)
        fig_rho = go.Figure()
        fig_rho.add_trace(go.Scatter(
            x=doy_range, y=rho_fit, mode="lines", name="\u03c1 (kg/m\u00b3)",
            line=dict(color="#9467bd", width=2),
        ))
        fig_rho.update_layout(
            xaxis_title="Day of Year", yaxis_title="Air Density (kg/m\u00b3)",
            margin=dict(t=20, b=40), height=320,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_rho, use_container_width=True)

# ── Active turbine datasource (set after tab_settings executes) ───────────────
turbine_df_active = turbine_fixed_df if use_fixed_data else turbine_df

# ── Daily ─────────────────────────────────────────────────────────────────────
with tab_daily:
    selected_date = st.date_input(
        "Select a day in 2017",
        value=datetime.date(2017, 1, 1),
        min_value=datetime.date(2017, 1, 1),
        max_value=datetime.date(2017, 12, 31),
        key="daily_date",
    )

    day_demand = demand_df[demand_df["Timestamp"].dt.date == selected_date]
    has_turbine = selected_date <= TURBINE_CUTOFF

    if has_turbine:
        day_turbine = turbine_df_active[turbine_df_active["Timestamp"].dt.date == selected_date]
        combined = build_merged(
            day_demand, day_turbine,
            rotor_diameter_m, rated_power_kw, curtail_on_potential,
            lut_wind, lut_power, rated_wind_speed_fit, cut_in_speed, cut_out_speed,
            cp_factor=cp_factor, availability_factor=availability_factor,
        )
        times = combined["Timestamp"].dt.strftime("%H:%M").tolist()

        fig = go.Figure()
        # Demand
        fig.add_trace(go.Scatter(
            x=times, y=combined["demand_kw"], name="Demand",
            mode="lines", line=dict(color="#1f77b4", width=2),
            hovertemplate="%{x}<br>Demand: %{y:,.0f} kW<extra></extra>",
        ))
        # Shading between demand and turbine power (surplus/deficit)
        fig.add_trace(go.Scatter(
            x=times, y=combined["actual_kw"], name="Turbine Power",
            mode="lines", line=dict(color="#ff7f0e", width=2),
            fill="tonexty", fillcolor="rgba(255,127,14,0.12)",
            hovertemplate="%{x}<br>Turbine: %{y:,.0f} kW<extra></extra>",
        ))
        # Potential power
        fig.add_trace(go.Scatter(
            x=times, y=combined["potential_kw"], name="Potential Power",
            mode="lines", line=dict(color="#aec7e8", width=1.5, dash="dash"),
            hovertemplate="%{x}<br>Potential: %{y:,.0f} kW<extra></extra>",
        ))
        # Curtailment
        fig.add_trace(go.Scatter(
            x=times, y=combined["curtailed_kw"], name="Curtailed",
            mode="lines", line=dict(color="#189e07", width=2, dash="dashdot"),
            hovertemplate="%{x}<br>Curtailed: %{y:,.0f} kW<extra></extra>",
        ))
        # Storm shutdown
        fig.add_trace(go.Scatter(
            x=times, y=combined["storm_shutdown_kw"], name="Storm Shutdown",
            mode="lines", line=dict(color="#f4370d", width=1.5, dash="dot"),
            hovertemplate="%{x}<br>Storm Shutdown: %{y:,.0f} kW<extra></extra>",
        ))
        # Unmet demand shading
        deficit_y = combined[["demand_kw", "actual_kw"]].min(axis=1)
        fig.add_trace(go.Scatter(
            x=times, y=deficit_y, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=times, y=combined["demand_kw"], name="Unmet Demand",
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(31,119,180,0.12)",
            hoverinfo="skip",
        ))
        # Wind speed on secondary axis
        fig.add_trace(go.Scatter(
            x=times, y=combined["Wind_ms"], name="Wind Speed",
            mode="lines", line=dict(color="#9467bd", width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="%{x}<br>Wind: %{y:.1f} m/s<extra></extra>",
        ))
        fig.update_layout(
            title=f"Demand vs Supply - {selected_date.strftime('%d %B %Y')}",
            xaxis_title="Time",
            yaxis=dict(title="Power (kW)"),
            yaxis2=dict(title="Wind Speed (m/s)", overlaying="y", side="right",
                        showgrid=False, rangemode="tozero"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=80),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tip: click a trace name in the legend to hide/show it. Double-click to isolate.")

        st.subheader("Daily Summary")
        _td  = combined["demand_kw"].sum() * 0.5
        _ta  = combined["actual_kw"].sum() * 0.5
        _tp  = combined["potential_kw"].sum() * 0.5
        _tc  = combined["curtailed_kw"].sum() * 0.5
        _te  = combined["export_kw"].sum() * 0.5
        _tu  = combined["unmet_kw"].sum() * 0.5
        _ch  = (combined["curtailed_kw"] > curtail_threshold_kw).sum() * 0.5
        _sh  = (combined["storm_shutdown_kw"] > 0).sum() * 0.5
        _cp  = 100 * _tc / _tp if _tp > 0 else 0.0
        _up  = 100 * _tu / _td if _td > 0 else 0.0
        _caf = 100 * _ta / _tp if _tp > 0 else 0.0
        _ss  = 100 * (_td - _tu) / _td if _td > 0 else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Peak Demand",        f"{combined['demand_kw'].max():,.0f} kW",
                    help="Highest 30-min average household demand recorded on this day.")
        col2.metric("Peak Turbine Power", f"{combined['actual_kw'].max():,.0f} kW",
                    help="Highest 30-min average turbine output on this day, after availability factor.")
        col3.metric("Total Demand",       f"{_td:,.0f} kWh",
                    help="Total energy consumed by all households over the 24-hour period.")
        col4.metric("Total Potential",    f"{_tp:,.0f} kWh",
                    help="Total theoretical wind energy available: Cₚ × ½ρAv³ × turbines. "
                         "Zeroed during storm shutdown intervals.")

        col1b, col2b, col3b = st.columns(3)
        col1b.metric("Total Turbine Power", f"{_ta:,.0f} kWh",
                     delta=f"{_caf:.1f}% of potential",
                     help="Total energy delivered by turbines today, after availability factor. "
                          "Delta shows this as a percentage of Potential Power.")
        col2b.metric("Total Exported",      f"{_te:,.0f} kWh",
                     help="Energy exported to the grid (surplus above demand, capped at 40,000 kW). "
                          "Only non-zero when the Export toggle is ON.")
        col3b.metric("Total Unmet Demand",  f"{_tu:,.0f} kWh",
                     delta=f"{_up:.1f}% of demand", delta_color="inverse",
                     help="Energy demand that turbines could not cover today. "
                          "Occurs during low wind, unavailability, or storm shutdown.")

        col4b, col5b, col6b, col7b = st.columns(4)
        col4b.metric("Total Curtailed",    f"{_tc:,.0f} kWh",
                     delta=f"{_cp:.1f}% of potential", delta_color="inverse",
                     help="Turbine surplus that could not be used or exported today. "
                          "Curtailment = max(0, actual − demand − export).")
        col5b.metric("Self-Sufficiency",   f"{_ss:.1f}%",
                     help="Percentage of today's demand met by local turbine generation. "
                          "100% means wind covered all demand at every interval.")
        col6b.metric("Curtailment Hours",  f"{_ch:.1f} h",
                     help=f"Hours today where curtailed power exceeded {curtail_threshold_kw:,} kW. "
                          f"Each qualifying 30-min interval counts as 0.5 h.")
        col7b.metric("Storm Shutdown Hours", f"{_sh:.1f} h",
                     help="Hours today where wind exceeded cut-out speed, forcing turbine shutdown. "
                          "Distinct from curtailment — turbines were physically off.")

    else:
        st.warning(
            f"Turbine data is only available up to {TURBINE_CUTOFF.strftime('%d %b %Y')}. "
            "Supply and curtailed energy cannot be shown for the selected date."
        )
        d = day_demand.copy()
        d["demand_kw"] = d["Demand_mean_kw"] * num_households
        times = d["Timestamp"].dt.strftime("%H:%M").tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=d["demand_kw"], name="Demand", mode="lines",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="%{x}<br>Demand: %{y:,.0f} kW<extra></extra>",
        ))
        fig.update_layout(
            title=f"Demand - {selected_date.strftime('%d %B %Y')}",
            xaxis_title="Time", yaxis_title="Power (kW)",
            hovermode="x unified", margin=dict(t=80),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Daily Summary")
        col1, col2 = st.columns(2)
        col1.metric("Peak Demand",  f"{d['demand_kw'].max():,.0f} kW")
        col2.metric("Total Demand", f"{d['demand_kw'].sum() * 0.5:,.0f} kWh")

# ── Monthly ───────────────────────────────────────────────────────────────────
with tab_monthly:
    month_name = st.selectbox("Select month", list(MONTHS.values()), key="month_sel")
    month_num = next(k for k, v in MONTHS.items() if v == month_name)

    d_slice = demand_df[demand_df["Timestamp"].dt.month == month_num]
    t_slice = turbine_df_active[turbine_df_active["Timestamp"].dt.month == month_num]
    totals = daily_totals(d_slice, t_slice,
                          rotor_diameter_m, rated_power_kw, curtail_on_potential,
                          lut_wind, lut_power, rated_wind_speed_fit, cut_in_speed, cut_out_speed,
                          cp_factor=cp_factor, availability_factor=availability_factor)

    x_labels = [d.strftime("%d %b") for d in totals.index]
    fig = make_figure(
        x_labels,
        totals["demand_kwh"], totals["potential_kwh"],
        totals["actual_kwh"], totals["curtailed_kwh"],
        f"Daily Energy Totals - {month_name} 2017", "kWh",
        storm_shutdown=totals["storm_shutdown_kwh"],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tip: click a trace name in the legend to hide/show it. Double-click to isolate.")

    st.subheader(f"{month_name} Summary")
    show_metrics(
        totals["demand_kwh"].sum(), totals["actual_kwh"].sum(),
        totals["potential_kwh"].sum(), totals["curtailed_kwh"].sum(),
        totals["export_kwh"].sum(), totals["unmet_kwh"].sum(),
        totals["curtailed_hours"].sum(),
        total_storm_hrs=totals["storm_shutdown_hours"].sum(),
    )

# ── Seasonal ──────────────────────────────────────────────────────────────────
with tab_seasonal:
    season_name = st.selectbox("Select season", list(SEASONS.keys()), key="season_sel")
    season_months = SEASONS[season_name]

    d_slice = demand_df[demand_df["Timestamp"].dt.month.isin(season_months)]
    t_slice = turbine_df_active[turbine_df_active["Timestamp"].dt.month.isin(season_months)]
    totals = daily_totals(d_slice, t_slice,
                          rotor_diameter_m, rated_power_kw, curtail_on_potential,
                          lut_wind, lut_power, rated_wind_speed_fit, cut_in_speed, cut_out_speed,
                          cp_factor=cp_factor, availability_factor=availability_factor)

    if season_months == [12, 1, 2]:
        jan_feb = totals[totals.index.month.isin([1, 2])]
        dec     = totals[totals.index.month == 12]
        x_seasonal = (
            [d.strftime("%d %b") for d in jan_feb.index]
            + [""]
            + [d.strftime("%d %b") for d in dec.index]
        )
        def _pad(col):
            return list(jan_feb[col]) + [None] + list(dec[col])
        fig = make_figure(
            x_seasonal,
            _pad("demand_kwh"), _pad("potential_kwh"),
            _pad("actual_kwh"), _pad("curtailed_kwh"),
            f"Daily Energy Totals - {season_name} 2017", "kWh",
            storm_shutdown=_pad("storm_shutdown_kwh"),
        )
        tick_indices = [i for i in range(0, len(x_seasonal), 7) if x_seasonal[i] != ""]
        fig.update_xaxes(
            tickmode="array",
            tickvals=[x_seasonal[i] for i in tick_indices],
            ticktext=[x_seasonal[i] for i in tick_indices],
        )
        totals = pd.concat([jan_feb, dec])
    else:
        fig = make_figure(
            totals.index,
            totals["demand_kwh"], totals["potential_kwh"],
            totals["actual_kwh"], totals["curtailed_kwh"],
            f"Daily Energy Totals - {season_name} 2017", "kWh",
            storm_shutdown=totals["storm_shutdown_kwh"],
        )
        week_ticks = pd.date_range(totals.index.min(), totals.index.max(), freq="W-MON")
        fig.update_xaxes(
            tickmode="array",
            tickvals=week_ticks,
            ticktext=[d.strftime("%d %b") for d in week_ticks],
        )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tip: click a trace name in the legend to hide/show it. Double-click to isolate.")

    st.subheader(f"{season_name} Summary")
    show_metrics(
        totals["demand_kwh"].sum(), totals["actual_kwh"].sum(),
        totals["potential_kwh"].sum(), totals["curtailed_kwh"].sum(),
        totals["export_kwh"].sum(), totals["unmet_kwh"].sum(),
        totals["curtailed_hours"].sum(),
        total_storm_hrs=totals["storm_shutdown_hours"].sum(),
    )

# ── Yearly ────────────────────────────────────────────────────────────────────
with tab_yearly:
    totals = daily_totals(demand_df, turbine_df_active,
                          rotor_diameter_m, rated_power_kw, curtail_on_potential,
                          lut_wind, lut_power, rated_wind_speed_fit, cut_in_speed, cut_out_speed,
                          cp_factor=cp_factor, availability_factor=availability_factor)

    fig = make_figure(
        totals.index,
        totals["demand_kwh"], totals["potential_kwh"],
        totals["actual_kwh"], totals["curtailed_kwh"],
        "Daily Energy Totals - 2017", "kWh",
        storm_shutdown=totals["storm_shutdown_kwh"],
    )
    month_ticks = pd.date_range("2017-01-01", "2017-12-01", freq="MS")
    fig.update_xaxes(
        tickmode="array",
        tickvals=month_ticks,
        ticktext=[d.strftime("%b") for d in month_ticks],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tip: click a trace name in the legend to hide/show it. Double-click to isolate.")

    st.subheader("Yearly Summary")
    show_metrics(
        totals["demand_kwh"].sum(), totals["actual_kwh"].sum(),
        totals["potential_kwh"].sum(), totals["curtailed_kwh"].sum(),
        totals["export_kwh"].sum(), totals["unmet_kwh"].sum(),
        totals["curtailed_hours"].sum(),
        total_storm_hrs=totals["storm_shutdown_hours"].sum(),
    )
