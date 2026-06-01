"""Generate plots for the compilation and gradient benchmark results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
COMPILATION_RESULTS = ROOT / "compilation_results.csv"
GRADIENT_RESULTS = ROOT / "gradient_results.csv"
OUTPUT_DIR = ROOT / "plots"

QISKIT_COLOR = "#2563eb"
PENNYLANE_COLOR = "#dc2626"


def load_results(path):
    """Load a benchmark CSV and fail clearly if its schema is incomplete."""
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark results: {path}")
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def save_figure(fig, filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_compilation_time(results):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        results["qubits"],
        results["qiskit_s"],
        marker="o",
        color=QISKIT_COLOR,
        label="Qiskit transpile",
    )
    ax.plot(
        results["qubits"],
        results["catalyst_s"],
        marker="s",
        color=PENNYLANE_COLOR,
        label="PennyLane Catalyst qjit",
    )
    ax.set(
        title="Compilation Time by Circuit Width",
        xlabel="Number of qubits",
        ylabel="Median compilation time (s)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "compilation_time.png")


def plot_compilation_memory(results):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        results["qubits"],
        results["qiskit_pyalloc_mb"],
        marker="o",
        color=QISKIT_COLOR,
        label="Qiskit transpile",
    )
    ax.plot(
        results["qubits"],
        results["catalyst_pyalloc_mb"],
        marker="s",
        color=PENNYLANE_COLOR,
        label="PennyLane Catalyst qjit",
    )
    ax.set(
        title="Compilation Python-Visible Peak Allocation",
        xlabel="Number of qubits",
        ylabel="Peak Python heap allocation (MB)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "compilation_memory.png")


def plot_compilation_depth(results):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        results["qubits"],
        results["qiskit_depth"],
        marker="o",
        color=QISKIT_COLOR,
        label="Qiskit transpile",
    )
    ax.plot(
        results["qubits"],
        results["catalyst_depth"],
        marker="s",
        color=PENNYLANE_COLOR,
        label="PennyLane specs before qjit",
    )
    ax.set(
        title="Compilation Benchmark Circuit Depth",
        xlabel="Number of qubits",
        ylabel="Circuit depth",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "compilation_depth.png")


def plot_gradient_time(results):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(
        results["num_params"],
        results["qiskit_mean_s"],
        yerr=results["qiskit_std_s"],
        marker="o",
        capsize=3,
        color=QISKIT_COLOR,
        label="Qiskit parameter-shift",
    )
    ax.errorbar(
        results["num_params"],
        results["pennylane_mean_s"],
        yerr=results["pennylane_std_s"],
        marker="s",
        capsize=3,
        color=PENNYLANE_COLOR,
        label="PennyLane adjoint",
    )
    ax.set(
        title="Gradient Computation Time by Parameter Count",
        xlabel="Number of parameters",
        ylabel="Mean gradient time (s, log scale)",
        yscale="log",
    )
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    save_figure(fig, "gradient_time.png")


def plot_gradient_memory(results):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        results["num_params"],
        results["qiskit_mem_mb"],
        marker="o",
        color=QISKIT_COLOR,
        label="Qiskit parameter-shift",
    )
    ax.plot(
        results["num_params"],
        results["pennylane_mem_mb"],
        marker="s",
        color=PENNYLANE_COLOR,
        label="PennyLane adjoint",
    )
    ax.set(
        title="Gradient Peak RSS Increase",
        xlabel="Number of parameters",
        ylabel="Peak RSS increase (MB)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "gradient_memory.png")


def main():
    compilation_results = load_results(COMPILATION_RESULTS)
    gradient_results = load_results(GRADIENT_RESULTS)

    plot_compilation_time(compilation_results)
    plot_compilation_memory(compilation_results)
    plot_compilation_depth(compilation_results)
    plot_gradient_time(gradient_results)
    plot_gradient_memory(gradient_results)

    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
