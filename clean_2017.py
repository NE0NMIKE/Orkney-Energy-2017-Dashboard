import csv
import os
import pandas as pd

base = os.path.dirname(os.path.abspath(__file__))

# ── 1. Residential demand: filter to 2017 ────────────────────────────────────
res_in  = os.path.join(base, "..", "Supplied Data", "Residential demand.csv")
res_out = os.path.join(base, "Residential_demand_2017.csv")

kept_res = 0
with open(res_in, newline="", encoding="utf-8") as fin, \
     open(res_out, "w", newline="", encoding="utf-8") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        if row and row[0].startswith("2017"):
            writer.writerow(row)
            kept_res += 1

print(f"Residential demand 2017: {kept_res} rows written -> {res_out}")

# ── 2. Turbine telemetry: filter to 2017, resample to 30-min intervals ───────
# Source format: DD/MM/YYYY HH:MM (1-min intervals)
# Aggregation: Power_kw -> mean, Wind_ms -> mean, Setpoint_kw -> mode
tur_in  = os.path.join(base, "..", "Supplied Data", "Turbine_telemetry.csv")
tur_out = os.path.join(base, "Turbine_telemetry_2017.csv")

df = pd.read_csv(
    tur_in,
    usecols=[0, 1, 2, 3],
    names=["Timestamp", "Power_kw", "Setpoint_kw", "Wind_ms"],
    header=0,
    parse_dates=["Timestamp"],
    dayfirst=True,
)

df = df[df["Timestamp"].dt.year == 2017].copy()
df = df.set_index("Timestamp")

df_30 = df.resample("30min").agg(
    Power_kw=("Power_kw", "mean"),
    Setpoint_kw=("Setpoint_kw", lambda x: x.mode().iloc[0] if len(x) > 0 else float("nan")),
    Wind_ms=("Wind_ms", "mean"),
)

# Cap the Power_kw to be always <= Setpoint_kw
df_30["Power_kw"] = df_30["Power_kw"].clip(upper=df_30["Setpoint_kw"])

df_30.to_csv(tur_out, float_format="%.4f")
print(f"Turbine telemetry 2017:  {len(df_30)} half-hour rows written -> {tur_out}")
