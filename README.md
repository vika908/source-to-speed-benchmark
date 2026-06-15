# Benchmark of Qiskit vs PennyLane Gradient Pipelines

This repository benchmarks gradient computation for quantum machine learning
circuits. The experiment fixes the circuit width at four qubits and increases
the number of variational parameters by changing the number of repeated
EfficientSU2-style layers.

The comparison focuses on three gradient pipelines:

- Qiskit parameter-shift with `ParamShiftEstimatorGradient`
- PennyLane `lightning.qubit` with adjoint differentiation
- PennyLane `lightning.qubit` with parameter-shift differentiation

The key distinction is that parameter-shift evaluates shifted circuits for each
trainable parameter, while adjoint differentiation computes the gradient more
directly for simulator-backed workflows. This makes the benchmark useful for
showing how gradient method choice affects scaling as parameter count grows.

## Circuit Equivalence and Validation

All three benchmark configurations use a four-qubit variational circuit with
`RY` and `RZ` rotations on each qubit, followed by an open linear CNOT chain.
For `reps = R`, the circuit has `R + 1` rotation layers and `R` entangling
layers. Each entangling layer applies:

```text
CNOT(0, 1), CNOT(1, 2), CNOT(2, 3)
```

The chain is open, not circular; no `CNOT(3, 0)` is used. The observable is
`Z^{\otimes 4}` in all configurations: PennyLane constructs it as a tensor
product of `PauliZ` operators, and Qiskit uses `SparsePauliOp("ZZZZ")`.

The PennyLane ansatz is written explicitly in
`gradient_pennylane_param_shift.py` and `gradient_pennylane_adjoint.py`. Qiskit
uses `EfficientSU2(num_qubits, reps=reps, entanglement="linear")`, so
`gradient_qiskit_param_shift.py` validates the decomposed Qiskit circuit before
timing starts. The function `validate_efficientsu2_structure(...)` checks that:

- the decomposed circuit contains only `ry`, `rz`, and `cx` gates
- each qubit has exactly `reps + 1` `RY` gates and `reps + 1` `RZ` gates
- the only CNOT edges are nearest-neighbor linear edges
- each expected edge, `(0, 1)`, `(1, 2)`, and `(2, 3)`, appears exactly `reps`
  times

This structural validation is part of the Qiskit benchmark code. It verifies
the rotation-layer contents and open-chain entanglement pattern, but it does not
by itself compare numerical expectation values or gradient entries.

For a numerical equivalence check, run:

```bash
venv/bin/python validate_circuit_equivalence.py
```

This script compares Qiskit parameter-shift against the PennyLane
parameter-shift circuit for small cases (`reps = 1` and `reps = 2` by default).
Because PennyLane stores parameters as interleaved `RY, RZ` pairs per qubit,
while Qiskit groups all `RY` parameters before all `RZ` parameters within each
rotation layer, the script applies a deterministic parameter-order mapping
before comparison. It then checks:

- equality of expectation values for `Z^{\otimes 4}`
- equality of gradient-vector length, expected to be `2 * NUM_QUBITS * (reps + 1)`
- maximum absolute difference between corresponding gradient entries

The default tolerance is `1e-10`. On the recorded validation run, the output was:

```text
Reps   | Params   | Exp diff     | Grad len | Grad max diff
----------------------------------------------------------------
1      | 16       | 1.110e-16    | 16       | 1.665e-16
2      | 24       | 1.110e-16    | 24       | 3.816e-16
```

Additional repetition counts can be checked with, for example:

```bash
venv/bin/python validate_circuit_equivalence.py --reps 1 2 3 --tolerance 1e-10
```

## Benchmark Scripts

These scripts run one gradient pipeline per process. They are the preferred
targets for process-level memory measurements with macOS `/usr/bin/time -l`.

Run with:

```bash
/usr/bin/time -l venv/bin/python gradient_qiskit_param_shift.py 2> time_output_qiskit.txt
/usr/bin/time -l venv/bin/python gradient_pennylane_adjoint.py 2> time_output_adjoint.txt
/usr/bin/time -l venv/bin/python gradient_pennylane_param_shift.py 2> time_output_pennylane_parameter.txt
```

Outputs:

- `gradient_qiskit_param_shift_results.csv`
- `gradient_pennylane_adjoint_results.csv`
- `gradient_pennylane_param_shift_results.csv`
- `time_output_qiskit.txt`
- `time_output_adjoint.txt`
- `time_output_pennylane_parameter.txt`

Timing metrics in the CSV files:

- `num_qubits`: fixed at four qubits
- `reps`: number of repeated variational layers
- `num_params`: total number of trainable parameters
- `*_mean_s`: mean gradient computation time after warm-up
- `*_std_s`: standard deviation of gradient computation time after warm-up

Memory metrics in the `/usr/bin/time -l` output files:

- `maximum resident set size`: peak resident memory for the complete script process
- `peak memory footprint`: macOS peak memory footprint for the complete script process
- `real`, `user`, `sys`: wall-clock, user CPU, and system CPU time for the complete script process

The memory values are process-level measurements for the whole script run,
including imports, warm-up, all parameter-count cases, and CSV writing. They are
not per-row memory measurements.

## Plotting

`plot.py` reads the three single-pipeline timing CSV files and the three
`/usr/bin/time -l` output files from the repository root.

Run with:

```bash
venv/bin/python plot.py
```

Outputs:

- `plots/gradient_time.png`: mean gradient time versus number of parameters, with standard-deviation error bars
- `plots/gradient_memory.png`: process-level memory comparison from `/usr/bin/time -l`

The plotting script only generates gradient benchmark graphs.
