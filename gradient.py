import csv
import time
import numpy as np
import tracemalloc
import os
import psutil
import threading

# ==========================================================
# Memory helper
# ==========================================================
process = psutil.Process(os.getpid())

def measure_peak(func):
    peak = 0
    stop = False

    def monitor():
        nonlocal peak
        while not stop:
            peak = max(peak, process.memory_info().rss)
            time.sleep(0.001)

    t = threading.Thread(target=monitor)
    t.start()

    result = func()

    stop = True
    t.join()

    return result, peak / (1024**2)

import pennylane as qml
from pennylane import numpy as pnp

from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient


# ==========================================================
# QISKIT: Parameter-Shift Gradient Benchmark
# ==========================================================
def benchmark_qiskit(num_qubits, reps, params, n_trials=5):
    try:
        ansatz = EfficientSU2(num_qubits, reps=reps, entanglement="linear")
        observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])

        estimator = Estimator()
        gradient = ParamShiftEstimatorGradient(estimator)

        # ---- warm-up (JIT / cache effects) ----
        _ = gradient.run([ansatz], [observable], [params]).result()

        baseline = process.memory_info().rss / (1024**2)

        def run_trials():
            times = []
            for _ in range(n_trials):
                start = time.perf_counter()
                _ = gradient.run([ansatz], [observable], [params]).result()
                times.append(time.perf_counter() - start)
            return times

        times, peak_mb = measure_peak(run_trials)
        mem_diff = peak_mb - baseline

        return float(np.mean(times)), float(np.std(times)), float(mem_diff)

    except Exception as e:
        return f"FAIL: {str(e)[:20]}", None, None


# ==========================================================
# PENNYLANE: Adjoint Gradient Benchmark (native)
# ==========================================================

def benchmark_pennylane(num_qubits, reps, params, n_trials=5):
    try:
        dev = qml.device("lightning.qubit", wires=num_qubits)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit(weights):
            idx = 0

            # initial layer
            for i in range(num_qubits):
                qml.RY(weights[idx], wires=i)
                qml.RZ(weights[idx + 1], wires=i)
                idx += 2

            # variational layers
            for _ in range(reps):
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

                for i in range(num_qubits):
                    qml.RY(weights[idx], wires=i)
                    qml.RZ(weights[idx + 1], wires=i)
                    idx += 2

            obs = qml.PauliZ(0)
            for i in range(1, num_qubits):
                obs = obs @ qml.PauliZ(i)

            return qml.expval(obs)

        grad_fn = qml.grad(circuit)

        # ---- warm-up (compile / cache) ----
        # Measure the baseline BEFORE the warm-up
        baseline = process.memory_info().rss / (1024**2)

        # ---- warm-up (compile / cache) ----
        _ = grad_fn(params)

        def run_trials():
            times = []
            for _ in range(n_trials):
                start = time.perf_counter()
                _ = grad_fn(params)
                times.append(time.perf_counter() - start)
            return times

        times, peak_mb = measure_peak(run_trials)
        mem_diff = peak_mb - baseline

        return float(np.mean(times)), float(np.std(times)), float(mem_diff)

    except Exception as e:
        return f"FAIL: {str(e)[:20]}", None, None


# ==========================================================
# MAIN EXPERIMENT
# ==========================================================
if __name__ == "__main__":

    NUM_QUBITS = 4
    REP_SCALES = [1, 10, 20, 40, 60, 80, 100]
    RESULTS_FILE = "profile_compilation.csv"

    results = []

    print("\n--- CLEAN GRADIENT BENCHMARK (Adjoint vs Parameter-Shift) ---")
    print(f"{'Params':<8} | {'Qiskit mean±std':<25} | {'PL mean±std':<25} | {'Q-Mem':<8} | {'PL-Mem'}")
    print("-" * 90)

    for reps in REP_SCALES:

        num_params = (reps + 1) * NUM_QUBITS * 2

        np.random.seed(42)
        base_params = np.random.uniform(-np.pi, np.pi, num_params)

        q_mean, q_std, q_mem = benchmark_qiskit(NUM_QUBITS, reps, base_params)
        pl_mean, pl_std, pl_mem = benchmark_pennylane(NUM_QUBITS, reps, pnp.array(base_params, requires_grad=True))

        results.append({
            "num_qubits": NUM_QUBITS,
            "reps": reps,
            "num_params": num_params,
            "qiskit_mean_s": q_mean,
            "qiskit_std_s": q_std,
            "qiskit_mem_mb": q_mem,
            "pennylane_mean_s": pl_mean,
            "pennylane_std_s": pl_std,
            "pennylane_mem_mb": pl_mem,
        })

        q_str = f"{q_mean:.4f}±{q_std:.4f}" if isinstance(q_mean, float) else str(q_mean)
        pl_str = f"{pl_mean:.4f}±{pl_std:.4f}" if isinstance(pl_mean, float) else str(pl_mean)
        q_mem_str = f"{q_mem:.2f}" if isinstance(q_mem, float) else str(q_mem)
        pl_mem_str = f"{pl_mem:.2f}" if isinstance(pl_mem, float) else str(pl_mem)

        print(f"{num_params:<8} | {q_str:<25} | {pl_str:<25} | {q_mem_str:<8} | {pl_mem_str}")

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {RESULTS_FILE}")