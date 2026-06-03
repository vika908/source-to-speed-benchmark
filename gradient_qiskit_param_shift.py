import csv
import time

import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient


NUM_QUBITS = 4
REP_SCALES = [1, 10, 20, 30, 40, 50, 60]
N_TRIALS = 5
RESULTS_FILE = "gradient_qiskit_param_shift_results.csv"


def make_params(reps):
    num_params = (reps + 1) * NUM_QUBITS * 2
    np.random.seed(42)
    return np.random.uniform(-np.pi, np.pi, num_params)


def benchmark(num_qubits, reps, params, n_trials=N_TRIALS):
    ansatz = EfficientSU2(num_qubits, reps=reps, entanglement="linear")
    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])

    estimator = Estimator()
    gradient = ParamShiftEstimatorGradient(estimator)

    _ = gradient.run([ansatz], [observable], [params]).result()

    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _ = gradient.run([ansatz], [observable], [params]).result()
        times.append(time.perf_counter() - start)

    return float(np.mean(times)), float(np.std(times))


if __name__ == "__main__":
    results = []

    print("\n--- Qiskit Parameter-Shift Gradient Benchmark ---")
    print(f"{'Params':<8} | {'Mean s':<12} | {'Std s':<12}")
    print("-" * 39)

    for reps in REP_SCALES:
        num_params = (reps + 1) * NUM_QUBITS * 2
        mean_s, std_s = benchmark(NUM_QUBITS, reps, make_params(reps))

        results.append(
            {
                "num_qubits": NUM_QUBITS,
                "reps": reps,
                "num_params": num_params,
                "qiskit_mean_s": mean_s,
                "qiskit_std_s": std_s,
            }
        )

        print(f"{num_params:<8} | {mean_s:<12.6f} | {std_s:<12.6f}")

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {RESULTS_FILE}")
