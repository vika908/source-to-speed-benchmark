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
