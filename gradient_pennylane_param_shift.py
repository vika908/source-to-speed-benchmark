import csv
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


NUM_QUBITS = 4
REP_SCALES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_TRIALS = 5
DIFF_METHOD = "parameter-shift"
RESULTS_FILE = "gradient_pennylane_param_shift_results.csv"


def make_params(reps):
    num_params = (reps + 1) * NUM_QUBITS * 2
    np.random.seed(42)
    base_params = np.random.uniform(-np.pi, np.pi, num_params)
    return pnp.array(base_params, requires_grad=True)


def benchmark(num_qubits, reps, params, n_trials=N_TRIALS):
    dev = qml.device("lightning.qubit", wires=num_qubits)

    @qml.qnode(dev, diff_method=DIFF_METHOD)
    def circuit(weights):
        idx = 0

        for i in range(num_qubits):
            qml.RY(weights[idx], wires=i)
            qml.RZ(weights[idx + 1], wires=i)
            idx += 2

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
    _ = grad_fn(params)

    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _ = grad_fn(params)
        times.append(time.perf_counter() - start)

    return float(np.mean(times)), float(np.std(times))


if __name__ == "__main__":
    results = []

    print("\n--- PennyLane Parameter-Shift Gradient Benchmark ---")
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
                "pennylane_param_shift_mean_s": mean_s,
                "pennylane_param_shift_std_s": std_s,
            }
        )

        print(f"{num_params:<8} | {mean_s:<12.6f} | {std_s:<12.6f}")

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {RESULTS_FILE}")
