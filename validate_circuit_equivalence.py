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
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient

from gradient_qiskit_param_shift import (
    AER_ESTIMATOR_OPTIONS,
    NUM_QUBITS,
    validate_efficientsu2_structure,
)


DEFAULT_REPS = [1, 2]
DEFAULT_TOLERANCE = 1e-10


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

    qiskit_ansatz = EfficientSU2(NUM_QUBITS, reps=reps, entanglement="linear")
    validate_efficientsu2_structure(qiskit_ansatz, NUM_QUBITS, reps)
    qiskit_ansatz = qiskit_ansatz.decompose()
    observable = SparsePauliOp.from_list([("Z" * NUM_QUBITS, 1)])
    estimator = EstimatorV2(options=AER_ESTIMATOR_OPTIONS)

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
