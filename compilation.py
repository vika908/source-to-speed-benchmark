import csv
import time
import numpy as np
import tracemalloc

import pennylane as qml
from catalyst import qjit
from qiskit.circuit.library import EfficientSU2
from qiskit import transpile


# ==========================================================
# Utility: stable timing
# ==========================================================
def timed(fn, warmup=2, repeats=5):
    for _ in range(warmup):
        fn()

    vals = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - start)

    return float(np.median(vals))


# ==========================================================
# QISKIT (With Depth Tracking)
# ==========================================================
def benchmark_qiskit(num_qubits, reps):
    try:
        circuit = EfficientSU2(num_qubits, reps=reps, entanglement="linear").decompose()

        # Run transpile once externally to evaluate the compiled depth metric
        transpiled_circuit = transpile(
            circuit,
            coupling_map=None,
            optimization_level=1,
            basis_gates=["rz", "sx", "x", "cx"]
        )
        final_depth = transpiled_circuit.depth()

        def compile_fn():
            transpile(
                circuit,
                coupling_map=None,
                optimization_level=1,
                basis_gates=["rz", "sx", "x", "cx"]
            )

        tracemalloc.start()
        t = timed(compile_fn)
        _, mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return t, mem / (1024 * 1024), final_depth

    except Exception as e:
        tracemalloc.stop()
        return f"FAIL: {str(e)[:20]}", "FAIL", "FAIL"


# ==========================================================
# CATALYST (With Depth Tracking via QNode Specs)
# ==========================================================
def benchmark_catalyst(num_qubits, reps):

    dev = qml.device("lightning.qubit", wires=num_qubits)

    def build_qnode():
        @qml.qnode(dev)
        def circuit():
            # layer 1: variable rotations
            for i in range(num_qubits):
                qml.RY(0.1 * (i + 1), wires=i)
                qml.RZ(0.2 * (num_qubits - i), wires=i)

            # entanglement pattern depends on qubit parity
            for layer in range(reps):
                for i in range(num_qubits - 1):
                    if (i + layer) % 2 == 0:
                        qml.CNOT(wires=[i, i + 1])
                    else:
                        qml.CZ(wires=[i, i + 1])

                # alternating rotation structure
                for i in range(num_qubits):
                    if i % 3 == 0:
                        qml.RX(0.3 * i, wires=i)
                    elif i % 3 == 1:
                        qml.RY(0.2 * i, wires=i)
                    else:
                        qml.RZ(0.1 * i, wires=i)

            return qml.expval(qml.PauliZ(0))

        return circuit

    circuit = build_qnode()

    # Query the underlying PennyLane graph engine for circuit specs before JIT
    # Query the underlying PennyLane graph engine for circuit specs before JIT
    try:
        specs_out = qml.specs(circuit)()
        
        # Handle PennyLane v0.30+ (dictionary with nested 'resources' object)
        if isinstance(specs_out, dict) and "resources" in specs_out:
            final_depth = specs_out["resources"].depth
            
        # Handle older PennyLane versions (flat dictionary)
        elif isinstance(specs_out, dict):
            final_depth = specs_out.get("depth", 0)
            
        # Handle latest PennyLane v0.40+ (returns a specs object directly)
        else:
            final_depth = specs_out.resources.depth
            
    except Exception as e:
        final_depth = f"ERR: {e}"

    def compile_fn():
        jit_fn = qjit(circuit)
        jit_fn.jit_compile(())

    tracemalloc.start()
    compile_time = timed(compile_fn)
    _, mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return compile_time, mem / (1024 * 1024), final_depth


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    QUBITS = [1, 4, 8, 12, 16, 20, 24, 28, 32]
    REPS = 2
    OUTPUT = "clean3_compilation_fixed_structure.csv"

    print("\n--- FIXED STRUCTURAL VARIATION BENCHMARK (WITH DEPTH) ---")
    print(f"{'Qubits':<6} | {'Q-Time':<9} | {'Cat-Time':<11} | {'Q-PyAlloc':<9} | {'Cat-PyAlloc':<11} | {'Q-Depth':<7} | {'Cat-Depth'}")
    print("-" * 88)

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qubits", "reps", "qiskit_s", "catalyst_s", "qiskit_pyalloc_mb", "catalyst_pyalloc_mb", "qiskit_depth", "catalyst_depth"])

        for n in QUBITS:
            q_t, q_m, q_d = benchmark_qiskit(n, REPS)
            c_t, c_m, c_d = benchmark_catalyst(n, REPS)

            writer.writerow([n, REPS, q_t, c_t, q_m, c_m, q_d, c_d])

            # Format outputs elegantly for terminal reading
            q_t_str = f"{q_t:.4f}s" if isinstance(q_t, float) else str(q_t)
            c_t_str = f"{c_t:.4f}s" if isinstance(c_t, float) else str(c_t)
            q_m_str = f"{q_m:.2f}" if isinstance(q_m, float) else str(q_m)
            c_m_str = f"{c_m:.2f}" if isinstance(c_m, float) else str(c_m)

            print(f"{n:<6} | {q_t_str:<9} | {c_t_str:<11} | {q_m_str:<9} | {c_m_str:<11} | {q_d:<7} | {c_d}")