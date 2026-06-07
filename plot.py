"""Generate plots for the gradient benchmark results."""

import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/source-to-speed-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "plots"

TIMING_FILES = {
    "Qiskit parameter-shift": (
        ROOT / "gradient_qiskit_param_shift_results.csv",
        "qiskit_mean_s",
        "qiskit_std_s",
    ),
    "PennyLane adjoint": (
        ROOT / "gradient_pennylane_adjoint_results.csv",
        "pennylane_adjoint_mean_s",
        "pennylane_adjoint_std_s",
    ),
    "PennyLane parameter-shift": (
        ROOT / "gradient_pennylane_param_shift_results.csv",
        "pennylane_param_shift_mean_s",
        "pennylane_param_shift_std_s",
    ),
}

MEMORY_FILES = {
    "Qiskit parameter-shift": ROOT / "time_qiskit_param_shift.txt",
    "PennyLane adjoint": ROOT / "time_pennylane_adjoint.txt",
    "PennyLane parameter-shift": ROOT / "time_pennylane_param_shift.txt",
}

COLORS = {
    "Qiskit parameter-shift": "#2563eb",
    "PennyLane adjoint": "#dc2626",
    "PennyLane parameter-shift": "#16a34a",
}

MARKERS = {
    "Qiskit parameter-shift": "o",
    "PennyLane adjoint": "s",
    "PennyLane parameter-shift": "^",
}

BYTES_PER_MIB = 1024**2


def load_timing_results(path, mean_column, std_column):
    if not path.exists():
        raise FileNotFoundError(f"Missing gradient timing results: {path}")

    timing_lines = []
    with path.open(newline="") as f:
        for line in f:
            if not line.strip():
                break
            timing_lines.append(line)

    rows = list(csv.DictReader(timing_lines))

    required = {"num_params", mean_column, std_column}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return {
        "num_params": [int(row["num_params"]) for row in rows],
        "mean_s": [float(row[mean_column]) for row in rows],
        "std_s": [float(row[std_column]) for row in rows],
    }


def parse_time_output(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing /usr/bin/time -l output: {path}")

    text = path.read_text()
    max_rss = re.search(r"(\d+)\s+maximum resident set size", text)
    peak_footprint = re.search(r"(\d+)\s+peak memory footprint", text)

    if not max_rss:
        raise ValueError(f"{path} does not contain maximum resident set size")
    if not peak_footprint:
        raise ValueError(f"{path} does not contain peak memory footprint")

    return {
        "max_rss_mib": int(max_rss.group(1)) / BYTES_PER_MIB,
        "peak_footprint_mib": int(peak_footprint.group(1)) / BYTES_PER_MIB,
    }


def save_figure(fig, filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_gradient_time(results):
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, values in results.items():
        ax.errorbar(
            values["num_params"],
            values["mean_s"],
            yerr=values["std_s"],
            marker=MARKERS[label],
            capsize=3,
            linewidth=2,
            color=COLORS[label],
            label=label,
        )

    ax.set(
        title="Gradient Computation Time",
        xlabel="Number of parameters",
        ylabel="Mean gradient time (s, log scale)",
        yscale="log",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    save_figure(fig, "gradient_time.png")


def plot_gradient_memory(results):
    labels = list(results.keys())
    x_positions = range(len(labels))
    width = 0.36

    max_rss = [results[label]["max_rss_mib"] for label in labels]
    peak_footprint = [results[label]["peak_footprint_mib"] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [x - width / 2 for x in x_positions],
        max_rss,
        width=width,
        color=[COLORS[label] for label in labels],
        label="Maximum resident set size",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        peak_footprint,
        width=width,
        color=[COLORS[label] for label in labels],
        alpha=0.45,
        hatch="//",
        label="Peak memory footprint",
    )

    ax.set(
        title="Gradient Process Memory",
        xlabel="Pipeline",
        ylabel="Memory (MiB)",
    )
    ax.set_xticks(list(x_positions), labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, "gradient_memory.png")


def main():
    timing_results = {
        label: load_timing_results(path, mean_column, std_column)
        for label, (path, mean_column, std_column) in TIMING_FILES.items()
    }
    memory_results = {
        label: parse_time_output(path) for label, path in MEMORY_FILES.items()
    }

    plot_gradient_time(timing_results)
    plot_gradient_memory(memory_results)

    print(f"Saved gradient plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
