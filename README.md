# Benchmark of Qiskit vs Pennylane with Quantum Machine Learning algorithms
The two frameworks possess fundamentally different standard gradient estimation methods. Qiskit's native method for estimating the gradient relies on the parameter-shift rule, while PennyLane relies heavily on adjoint differentiation. These two methods calculate the derivative of the cost function in fundamentally different ways: the parameter-shift rule scales linearly $O(M)$ with respect to the number of parameters, whereas adjoint differentiation scaling remains $O(1)$. Therefore, the most critical architectural differences between these two frameworks are exposed in algorithms with massive parameter counts. (More details in the LaTeX paper).   Considering this, I chose to benchmark these frameworks using Quantum Machine Learning (QML). QML is a family of algorithms designed for model training and is part of the broader group of Variational Quantum Algorithms (VQAs). The primary reason for selecting this family is that QML architectures require thousands of parameters to achieve sufficient data expressivity, making it perfect for demonstrating the scaling differences between Qiskit and PennyLane.  
In reality, a trained model is of no practical use if it remains as a logical circuit. One must deploy the model onto physical hardware to utilize it.

## Benchmark Scripts

### `compilation.py`
Benchmarks software compilation overhead as the number of qubits scales. For each qubit count, the script compares Qiskit's `transpile` against PennyLane Catalyst's `qjit`. It uses `REPS = 2` and tests 1, 4, 8, ..., 32 qubits.

> [!NOTE]
> Memory is measured using Python's `tracemalloc`, which captures Python heap allocations but does not fully account for native allocations in LLVM, MLIR, or C++ simulator backends. Therefore, memory results are interpreted as Python-level framework overhead rather than total process memory consumption.

**Frameworks**: Qiskit vs. PennyLane (Catalyst)

**Measured Metrics**:
- **Compilation Time (s)**: Median wall time over five measured runs after two warm-up runs. Qiskit transpiles to the `rz`, `sx`, `x`, and `cx` basis gates with optimization level 1. Catalyst creates a `qjit` wrapper and invokes `jit_compile`.
- **Python-Visible Peak Allocation (MB)**: Peak Python heap memory consumed during compilation. *Qiskit shows a small gradually increasing Python-level allocation, while Catalyst exhibits a larger but nearly constant Python-visible overhead. Because Catalyst relies on native compilation infrastructure, the reported memory is a lower bound and should not be interpreted as total compiler memory.*
- **Circuit Depth**: Qiskit's depth is taken from the transpiled circuit. Catalyst's depth is read from PennyLane `qml.specs` before JIT compilation, so the two values describe different stages of their respective pipelines.

Run with:
```bash
python compilation.py
```

Results are written to `clean3_compilation_fixed_structure.csv`.

---

### `gradient.py`
Benchmarks gradient computation as the number of parameters increases. The script fixes the circuit width at four qubits and tests 1, 10, 20, 40, 60, 80, and 100 repetitions. Qiskit uses `ParamShiftEstimatorGradient`, while PennyLane uses `lightning.qubit` with `diff_method="adjoint"`.

**Frameworks**: Qiskit (Parameter-Shift) vs. PennyLane (Adjoint AD)

**Measured Metrics**:
- **Execution Time (s)**: Mean and standard deviation of the time taken for the gradient computation after warm-up.
- **Peak RSS Increase (MB)**: Process resident-memory peak sampled during five measured gradient runs, minus a framework-specific baseline. Qiskit's baseline is taken after warm-up. PennyLane's baseline is taken before warm-up, so its value also includes any resident-memory growth caused by warm-up. This is process-level RSS rather than Python-only heap allocation.

Run with:
```bash
python gradient.py
```

Results are written to `profile_compilation.csv`.
