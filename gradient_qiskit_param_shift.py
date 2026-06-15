import csv
import time
from collections import Counter

import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient


NUM_QUBITS = 4
REP_SCALES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_TRIALS = 5
RESULTS_FILE = "gradient_qiskit_param_shift_results.csv"
AER_ESTIMATOR_OPTIONS = {
    "backend_options": {"method": "statevector"},
    "default_precision": 0.0,
}
ROTATION_GATES = {"ry", "rz"}
ENTANGLING_GATE = "cx"


def make_params(reps):
    num_params = (reps + 1) * NUM_QUBITS * 2
    np.random.seed(42)
    return np.random.uniform(-np.pi, np.pi, num_params)


def validate_efficientsu2_structure(ansatz, num_qubits, reps):
    decomposed = ansatz.decompose()
    allowed_gates = ROTATION_GATES | {ENTANGLING_GATE}
    rotation_counts = Counter()
    entangling_edges = []

    for instruction in decomposed.data:
        gate_name = instruction.operation.name
        qubits = tuple(decomposed.find_bit(qubit).index for qubit in instruction.qubits)

        if gate_name not in allowed_gates:
            raise ValueError(f"Unexpected gate in decomposed ansatz: {gate_name}")

        if gate_name in ROTATION_GATES:
            if len(qubits) != 1:
                raise ValueError(f"{gate_name} gate is not single-qubit: {qubits}")
            rotation_counts[(gate_name, qubits[0])] += 1
        else:
            entangling_edges.append(qubits)

    expected_rotation_count = reps + 1
    for gate_name in ROTATION_GATES:
        for qubit in range(num_qubits):
            actual_count = rotation_counts[(gate_name, qubit)]
            if actual_count != expected_rotation_count:
                raise ValueError(
                    f"Expected {expected_rotation_count} {gate_name} gates on qubit "
                    f"{qubit}, found {actual_count}"
                )

    expected_linear_edges = [(i, i + 1) for i in range(num_qubits - 1)]
    expected_edge_counts = Counter(
        edge for _ in range(reps) for edge in expected_linear_edges
    )
    actual_edge_counts = Counter(entangling_edges)
    if actual_edge_counts != expected_edge_counts:
        raise ValueError(
            "Decomposed ansatz does not match the expected linear CNOT pattern: "
            f"expected {dict(expected_edge_counts)}, found {dict(actual_edge_counts)}"
        )


def benchmark(num_qubits, reps, params, n_trials=N_TRIALS):
    ansatz = EfficientSU2(num_qubits, reps=reps, entanglement="linear")
    validate_efficientsu2_structure(ansatz, num_qubits, reps)
    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])

    estimator = AerEstimator(options=AER_ESTIMATOR_OPTIONS)
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
