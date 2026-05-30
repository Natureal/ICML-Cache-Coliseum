import re
import pandas as pd
import matplotlib.pyplot as plt


RAW_TABLE = r"""
+-------------------------------------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
|                    Name                   |   LogDis-0  |   LogDis-5  |  LogDis-10  |  LogDis-20  |  LogDis-30  |  LogDis-40  |  LogDis-50  |
+-------------------------------------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
|                    LRU                    | 0.638/1.886 | 0.638/1.886 | 0.638/1.886 | 0.638/1.886 | 0.638/1.886 | 0.638/1.886 | 0.638/1.886 |
|                   Marker                  | 0.631/1.924 | 0.631/1.924 | 0.631/1.924 | 0.631/1.924 | 0.631/1.924 | 0.631/1.924 | 0.631/1.924 |
|             OnlineMin-msf-1000            | 0.612/2.022 | 0.612/2.022 | 0.612/2.022 | 0.612/2.022 | 0.612/2.022 | 0.612/2.022 | 0.612/2.022 |
|              Predict[Belady]              | 0.808/1.000 | 0.714/1.491 | 0.651/1.820 | 0.616/2.000 | 0.588/2.149 | 0.589/2.144 | 0.585/2.165 |
|       CombDet[Predict[Belady], LRU]       | 0.808/1.000 | 0.706/1.532 | 0.646/1.845 | 0.632/1.916 | 0.630/1.930 | 0.631/1.922 | 0.631/1.924 |
|              PredMark[Belady]             | 0.740/1.355 | 0.681/1.660 | 0.660/1.771 | 0.654/1.805 | 0.649/1.828 | 0.647/1.841 | 0.646/1.847 |
|              LMarker[Belady]              | 0.740/1.355 | 0.660/1.770 | 0.640/1.874 | 0.637/1.892 | 0.633/1.912 | 0.636/1.899 | 0.633/1.912 |
|        FollowerRobust[OracleState]        | 0.808/1.000 | 0.652/1.814 | 0.628/1.938 | 0.618/1.991 | 0.616/2.000 | 0.615/2.006 | 0.616/2.002 |
|    PredictiveOnlineMin[Belady]-msf-100    | 0.808/1.000 | 0.645/1.850 | 0.614/2.010 | 0.613/2.017 | 0.615/2.007 | 0.615/2.006 | 0.614/2.012 |
|   RPB-new-OnlineMin[Belady]-msf-100-pb-0  | 0.808/1.000 | 0.649/1.830 | 0.619/1.985 | 0.615/2.009 | 0.619/1.984 | 0.614/2.013 | 0.615/2.008 |
|   RPB-new-OnlineMin[Belady]-msf-100-pb-1  | 0.808/1.000 | 0.680/1.665 | 0.635/1.903 | 0.621/1.974 | 0.624/1.958 | 0.623/1.967 | 0.619/1.986 |
|   RPB-new-OnlineMin[Belady]-msf-100-pb-2  | 0.808/1.000 | 0.695/1.600 | 0.652/1.830 | 0.636/1.902 | 0.631/1.910 | 0.628/1.920 | 0.634/1.895 |
"""

def parse_ascii_table(raw: str):
    header_line = None
    for line in raw.splitlines():
        if line.strip().startswith("|") and "LogDis" in line:
            header_line = line
            break
    if header_line is None:
        raise ValueError("Cannot find header line containing 'LogDis'.")

    header_parts = [p.strip() for p in header_line.strip().strip("|").split("|")]
    logdis = []
    for h in header_parts[1:]:
        m = re.search(r"LogDis-(\d+)", h)
        if not m:
            raise ValueError(f"Cannot parse LogDis value from header: {h}")
        logdis.append(int(m.group(1)))

    rows = []
    for line in raw.splitlines():
        if not line.strip().startswith("|"):
            continue
        if "Name" in line:
            continue

        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != 1 + len(logdis):
            continue

        name = re.sub(r"\s+", " ", parts[0]).strip()
        hits, crs = [], []
        for cell in parts[1:]:
            m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)\s*$", cell)
            if not m:
                hits.append(float("nan"))
                crs.append(float("nan"))
            else:
                hits.append(float(m.group(1)))
                crs.append(float(m.group(2)))
        rows.append((name, hits, crs))

    return logdis, rows


def rename_algo(name: str) -> str:
    """
    A general, extendable renaming function.
    - Exact match renames via RENAME_EXACT
    - Regex-based renames via RENAME_REGEX (applied in order)
    """
    RENAME_EXACT = {
        "Rand": "Random",
        "RDM-msf-20": "RDM",
        "Predict[Belady]": "BlindOracle",
        "OnlineMin-msf-1000": "OnlineMin",
        "PredictiveOnlineMin[Belady]-msf-100": r"OnOPT-OM",
        "RPB-new-OnlineMin[Belady]-msf-100-pb-0": r"RPB-OM ($\tau=0$)",
        "RPB-new-OnlineMin[Belady]-msf-100-pb-1": r"RPB-OM ($\tau=1$)",
        "RPB-new-OnlineMin[Belady]-msf-100-pb-2": r"RPB-OM ($\tau=2$)",
        "RPB-new-OnlineMin[Belady]-msf-100-pb-4": r"RPB-OM ($\tau=4$)",
        "RPB-new-RDM[Belady]-pb-0-msf-100": r"RPB-RDM ($\tau=0$)",
        "RPB-new-RDM[Belady]-pb-1-msf-100": r"RPB-RDM ($\tau=1$)",
        "RPB-new-RDM[Belady]-pb-2-msf-100": r"RPB-RDM ($\tau=2$)",
        "RPB-new-RDM[Belady]-pb-4-msf-100": r"RPB-RDM ($\tau=4$)",
        "PredictiveRDM[Belady]-msf-100": r"OnOPT-RDM",
        "FollowerRobust[OracleState]": "F&R",
        "PredMark[Belady]": "PredictiveMarker",
        "LMarker[Belady]": "LMarker",
        "LNonMarker[Belady]": "LNonMarker",
        "CombDet[Predict[Belady], Marker]": "BlindOracle&Marker",
        "CombDet[Predict[Belady], LRU]": "BlindOracle&LRU",
    }

    # (pattern, replacement) applied sequentially
    RENAME_REGEX = [
        # Example: uncomment if you want to shorten some long names later
        # (r"^RPB-new-OnlineMin\[Belady\]-msf-100-pb-(\d+)$", r"RPB-new(pb=\1)"),
    ]

    if name in RENAME_EXACT:
        return RENAME_EXACT[name]

    for pat, rep in RENAME_REGEX:
        if re.search(pat, name):
            return re.sub(pat, rep, name)

    return name


def main():
    logdis, rows = parse_ascii_table(RAW_TABLE)

    hit_cols = [f"Hit@{x}" for x in logdis]
    cr_cols = [f"CR@{x}" for x in logdis]

    records = []
    for name, hits, crs in rows:
        rec = {"Name": rename_algo(name)}  # apply rename here
        rec.update({c: v for c, v in zip(hit_cols, hits)})
        rec.update({c: v for c, v in zip(cr_cols, crs)})
        records.append(rec)
    df = pd.DataFrame(records)

        # Plot
    plt.figure(figsize=(8, 4.2))

    # Color-blind friendly palette (Okabe–Ito)
    PALETTE = {
        "black":  "#000000",
        "gray":   "#7F7F7F",
        "yellow": "#F0E442",
        "orange": "#E69F00",
	"red":    "#D55E00",
        "sky":    "#56B4E9",
        "green":  "#009E73",
        "blue":   "#0072B2",
        "verm":   "#D55E00",
        "purple": "#CC79A7",
    }

    BASELINES = {"LRU", "Marker", "Random", "RDM", "OnlineMin"}
    T_BASELINES = {"PredictiveMarker", "LMarker"}

    def is_rpb_om(n: str) -> bool:
        return n.startswith(r"RPB-OM")

    # Explicit colors AFTER renaming
    COLOR_MAP = {
        # Baselines (neutral)
        "LRU": PALETTE["gray"],
        "Marker": PALETTE["gray"],
        "Random": PALETTE["gray"],
        "RDM": PALETTE["gray"],
        "OnlineMin": PALETTE["gray"],
        "LMarker": PALETTE["orange"],
        "PredictiveMarker": PALETTE["purple"],
        "BlindOracle&LRU": PALETTE["yellow"],
        "F&R": PALETTE["green"],

        # Oracle / predictor related
        "BlindOracle": PALETTE["gray"],
        "OnOPT-OM": PALETTE["sky"],
        "OnOPT-RDM": PALETTE["yellow"],

        # RPB-OM family: one main highlight color
        # (all tau share the same color; distinguished by markers)
        r"RPB-OM ($\tau=0$)": PALETTE["blue"],
        r"RPB-OM ($\tau=1$)": PALETTE["orange"],
        r"RPB-OM ($\tau=2$)": PALETTE["verm"],
        r"RPB-OM ($\tau=4$)": PALETTE["verm"],
    }

    # Markers distinguish tau within the same method family
    MARKER_MAP = {
        # Baselines
        "LRU": "v", "Marker": "^", "Random": "o", "RDM": ".", "OnlineMin": "s",
        # Predictor / OnOPT
        "BlindOracle": "s", "OnOPT-OM": "D", "OnOPT-RDM": "D",
        # RPB-OM variants
        r"RPB-OM ($\tau=0$)": "o",
        r"RPB-OM ($\tau=1$)": "v",
        r"RPB-OM ($\tau=2$)": "^",
        r"RPB-OM ($\tau=4$)": "s",
    }

    # Optional: make RPB-OM slightly thicker to pop without being loud
    for _, r in df.iterrows():
        y = [r[f"CR@{x}"] for x in logdis][:5]
        name = r["Name"]

        color = COLOR_MAP.get(name, PALETTE["black"])
        ls = "--" if name in BASELINES else "-"
        if name in T_BASELINES:
            ls = "--"
        alpha = 0.6 if name in BASELINES else 1.0
        mk = MARKER_MAP.get(name, "o")

        lw = 1.5 if is_rpb_om(name) else (1.2 if name not in BASELINES else 1.2)
        ms = 4.5 if is_rpb_om(name) else (4.2 if name not in BASELINES else 4.2)

        plt.plot(
            logdis[:5], y,
            label=name,
            color=color,
            linestyle=ls,
            marker=mk,
            alpha=alpha,
            linewidth=lw,
            markersize=ms,
        )

    plt.xlabel("Noise level (the parameter of log-normal)", fontsize=14)
    plt.ylabel("Cost ratio", fontsize=14)
    plt.title("")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()

