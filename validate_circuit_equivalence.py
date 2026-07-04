"""Validate Qiskit/PennyLane circuit and gradient equivalence on small cases."""

import argparse
import os
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "source-to-speed-mpl"))
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient

try:
    from qiskit.circuit.library import efficient_su2
except ImportError:
    from qiskit.circuit.library import EfficientSU2

    def efficient_su2(
        num_qubits,
        su2_gates,
        entanglement,
        reps,
        skip_final_rotation_layer,
        skip_unentangled_qubits,
        insert_barriers,
    ):
        if su2_gates != ["ry", "rz"]:
            raise ValueError(f"Unsupported SU2 gate set: {su2_gates}")
        return EfficientSU2(
            num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            skip_unentangled_qubits=skip_unentangled_qubits,
            insert_barriers=insert_barriers,
        )


NUM_QUBITS = 4
DEFAULT_REPS = [1, 2]
DEFAULT_TOLERANCE = 1e-10
ROTATION_GATES = {"ry", "rz"}
ENTANGLING_GATE = "cx"


def make_pennylane_circuit(num_qubits, reps, diff_method):
    dev = qml.device("lightning.qubit", wires=num_qubits)

    @qml.qnode(dev, diff_method=diff_method)
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

    return circuit


def pennylane_to_qiskit_params(params, num_qubits, reps):
    qiskit_params = []
    idx = 0
    layers = []

    for _ in range(reps + 1):
        layer = []
        for _ in range(num_qubits):
            layer.append((params[idx], params[idx + 1]))
            idx += 2
        layers.append(layer)

    for layer in layers:
        qiskit_params.extend(ry for ry, _ in layer)
        qiskit_params.extend(rz for _, rz in layer)

    return np.array(qiskit_params)


def qiskit_to_pennylane_gradient(gradient, num_qubits, reps):
    pennylane_gradient = []
    idx = 0

    for _ in range(reps + 1):
        ry_values = gradient[idx : idx + num_qubits]
        rz_values = gradient[idx + num_qubits : idx + 2 * num_qubits]
        idx += 2 * num_qubits

        for qubit in range(num_qubits):
            pennylane_gradient.extend([ry_values[qubit], rz_values[qubit]])

    return np.array(pennylane_gradient)


def validate_efficientsu2_structure(ansatz, num_qubits, reps):
    allowed_gates = ROTATION_GATES | {ENTANGLING_GATE}
    rotation_counts = {}
    entangling_edges = []

    for instruction in ansatz.data:
        gate_name = instruction.operation.name
        qubits = tuple(ansatz.find_bit(qubit).index for qubit in instruction.qubits)

        if gate_name not in allowed_gates:
            raise ValueError(f"Unexpected gate in ansatz: {gate_name}")

        if gate_name in ROTATION_GATES:
            if len(qubits) != 1:
                raise ValueError(f"{gate_name} gate is not single-qubit: {qubits}")
            rotation_counts[(gate_name, qubits[0])] = (
                rotation_counts.get((gate_name, qubits[0]), 0) + 1
            )
        else:
            entangling_edges.append(qubits)

    expected_rotation_count = reps + 1
    for gate_name in ROTATION_GATES:
        for qubit in range(num_qubits):
            actual_count = rotation_counts.get((gate_name, qubit), 0)
            if actual_count != expected_rotation_count:
                raise ValueError(
                    f"Expected {expected_rotation_count} {gate_name} gates on qubit "
                    f"{qubit}, found {actual_count}"
                )

    expected_edges = [
        (control, control + 1)
        for _ in range(reps)
        for control in range(num_qubits - 1)
    ]
    if entangling_edges != expected_edges:
        raise ValueError(
            "Ansatz does not match the expected linear CNOT pattern: "
            f"expected {expected_edges}, found {entangling_edges}"
        )


def validate_case(reps, tolerance):
    num_params = 2 * NUM_QUBITS * (reps + 1)
    rng = np.random.default_rng(42 + reps)
    pennylane_params = rng.uniform(-np.pi, np.pi, num_params)
    qiskit_params = pennylane_to_qiskit_params(pennylane_params, NUM_QUBITS, reps)

    pennylane_circuit = make_pennylane_circuit(
        NUM_QUBITS, reps, diff_method="parameter-shift"
    )
    pennylane_params_grad = pnp.array(pennylane_params, requires_grad=True)
    pennylane_expval = float(pennylane_circuit(pennylane_params_grad))
    pennylane_gradient = np.array(qml.grad(pennylane_circuit)(pennylane_params_grad))

    qiskit_ansatz = efficient_su2(
        NUM_QUBITS,
        su2_gates=["ry", "rz"],
        entanglement="linear",
        reps=reps,
        skip_final_rotation_layer=False,
        skip_unentangled_qubits=False,
        insert_barriers=False,
    )
    validate_efficientsu2_structure(qiskit_ansatz, NUM_QUBITS, reps)
    observable = SparsePauliOp.from_list([("Z" * NUM_QUBITS, 1)])
    estimator = AerEstimatorV2(
        options={
            "backend_options": {"method": "statevector"},
            "run_options": {"shots": None},
        }
    )

    qiskit_expval = float(
        estimator.run([(qiskit_ansatz, observable, qiskit_params)]).result()[0].data.evs
    )
    qiskit_gradient = np.array(
        ParamShiftEstimatorGradient(estimator)
        .run([qiskit_ansatz], [observable], [qiskit_params])
        .result()
        .gradients[0]
    )
    qiskit_gradient = qiskit_to_pennylane_gradient(
        qiskit_gradient, NUM_QUBITS, reps
    )

    expval_diff = abs(pennylane_expval - qiskit_expval)
    gradient_max_diff = float(np.max(np.abs(pennylane_gradient - qiskit_gradient)))
    gradient_lengths_match = len(pennylane_gradient) == len(qiskit_gradient) == num_params

    if expval_diff > tolerance:
        raise AssertionError(
            f"Expectation values differ for reps={reps}: {expval_diff}"
        )
    if not gradient_lengths_match:
        raise AssertionError(
            f"Gradient length mismatch for reps={reps}: "
            f"{len(pennylane_gradient)} vs {len(qiskit_gradient)}"
        )
    if gradient_max_diff > tolerance:
        raise AssertionError(
            f"Gradient values differ for reps={reps}: {gradient_max_diff}"
        )

    return {
        "reps": reps,
        "num_params": num_params,
        "expval_diff": expval_diff,
        "gradient_length": len(pennylane_gradient),
        "gradient_max_diff": gradient_max_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", nargs="+", type=int, default=DEFAULT_REPS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    print("\n--- Circuit Equivalence Validation ---")
    print(f"Tolerance: {args.tolerance:.1e}")
    print(f"{'Reps':<6} | {'Params':<8} | {'Exp diff':<12} | {'Grad len':<8} | {'Grad max diff':<14}")
    print("-" * 64)

    for reps in args.reps:
        result = validate_case(reps, args.tolerance)
        print(
            f"{result['reps']:<6} | "
            f"{result['num_params']:<8} | "
            f"{result['expval_diff']:<12.3e} | "
            f"{result['gradient_length']:<8} | "
            f"{result['gradient_max_diff']:<14.3e}"
        )

    print("\nValidation passed.")


if __name__ == "__main__":
    main()
