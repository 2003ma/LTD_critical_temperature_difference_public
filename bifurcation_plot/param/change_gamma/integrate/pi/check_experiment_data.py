import numpy as np
from pathlib import Path
import pandas as pd

V0 = 44900e-9
s = 71e-6
r = 3.5e-3
I = 5.7e-5
P0 = 101300
n = 0.00185
R = 8.314
Gamma = I / 20
T_top = 24 + 273.15

# Read the CSV beside this script, independent of the working directory.
experiment_csv_path = Path(__file__).resolve().with_name("experiment_plot_include_L.csv")
experiment_df = pd.read_csv(experiment_csv_path)
no_load_experiment_ids = [
    "20200310-2",
    "20200310-3",
    "20200310-5",
    "20200311-2",
    "20200325-1",
    "20200325-2",
]


def first_stopped_delta_t(df, experiment_id, w_tol=1e-5):
    """Return the temperature difference at the first stopped point in CSV measurement order."""
    dT = pd.to_numeric(df[f"{experiment_id}-dT"], errors="raise")
    w = pd.to_numeric(df[f"{experiment_id}-w"], errors="raise")

    mask = dT.notna() & w.notna() & (w.abs() <= w_tol)
    stopped_dT = dT.loc[mask]

    if stopped_dT.empty:
        raise ValueError(f"{experiment_id}: 停止相当の点がありません")

    return float(stopped_dT.iloc[0])


experimental_data = [
    {
        "condition": "No added load",
        "DeltaT_c": float(np.mean([
            first_stopped_delta_t(experiment_df, experiment_id)
            for experiment_id in no_load_experiment_ids
        ])),
        "Gamma": I / 20.0,
    },
    {"condition": "L1", "DeltaT_c": first_stopped_delta_t(experiment_df, "loaded1"), "Gamma": I / 13.2},
    {"condition": "L2", "DeltaT_c": first_stopped_delta_t(experiment_df, "loaded2"), "Gamma": I / 9.1},
    {"condition": "L4", "DeltaT_c": first_stopped_delta_t(experiment_df, "loaded4"), "Gamma": I / 6.9},
]

# 元コードと同じ関数で選んだ臨界温度差だけを出力する。
report_lines = ["No load（平均に使った6点）:"]
for experiment_id in no_load_experiment_ids:
    value = first_stopped_delta_t(experiment_df, experiment_id)
    report_lines.append(f"{experiment_id}: {value:.10g} K")

report_lines.append("")
for row in experimental_data[1:]:
    report_lines.append(f"{row['condition']}: {row['DeltaT_c']:.10g} K")

report = "\n".join(report_lines) + "\n"
output_path = Path(__file__).resolve().with_suffix(".txt")
output_path.write_text(report, encoding="utf-8")
print(report, end="")

