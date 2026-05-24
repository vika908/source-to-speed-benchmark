import csv
import time
import tracemalloc
import numpy as np
from pennylane import numpy as pnp
import pennylane as qml

from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient

# Qiskit Benchmark
def benchmark_qiskit(num_qubits, reps, params):
    # Build the circuit and observable
    ansatz = EfficientSU2(num_qubits, reps=reps, entanglement='linear')
    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])
    
    # Setup the exact Estimator and Gradient method
    estimator = Estimator()
    gradient = ParamShiftEstimatorGradient(estimator)
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Calculate gradient
    result = gradient.run([ansatz], [observable], [params]).result()
    
    end_time = time.perf_counter()
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Time and memory
    wall_time = end_time - start_time
    peak_mb = peak_memory / (1024 * 1024)
    
    return wall_time, peak_mb

# PennyLane Benchmark
def benchmark_pennylane(num_qubits, reps, params):
    dev = qml.device("lightning.qubit", wires=num_qubits)
    
    # QNode with Adjoint Differentiation
    @qml.qnode(dev, diff_method="adjoint")

    # EfficientSU2 architecture
    def circuit(weights):
        param_idx = 0
        
        # Initial rotation layer
        for i in range(num_qubits):
            qml.RY(weights[param_idx], wires=i)
            qml.RZ(weights[param_idx + 1], wires=i)
            param_idx += 2
            
        # Entanglement and subsequent rotation layers
        for _ in range(reps):
            # Linear entanglement
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            # Rotation layer
            for i in range(num_qubits):
                qml.RY(weights[param_idx], wires=i)
                qml.RZ(weights[param_idx + 1], wires=i)
                param_idx += 2
                
        # Global Z observable
        obs = qml.PauliZ(0)
        for i in range(1, num_qubits):
            obs = obs @ qml.PauliZ(i)
        return qml.expval(obs)

    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Calculate gradient
    grad_fn = qml.grad(circuit)
    grad_result = grad_fn(params)
    
    end_time = time.perf_counter()
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Time and memory
    wall_time = end_time - start_time
    peak_mb = peak_memory / (1024 * 1024)
    
    return wall_time, peak_mb

if __name__ == "__main__":
    # Parameters
    NUM_QUBITS = 4
    REP_SCALES = [1, 10, 20, 50, 70, 100]
    RESULTS_FILE = "gradient_benchmark_results.csv"
    results = []
    
    for reps in REP_SCALES:
        # Number of parameters
        num_params = (reps + 1) * NUM_QUBITS * 2
        
        # Shared random starting parameters
        np.random.seed(42)
        initial_params = np.random.uniform(-np.pi, np.pi, num_params)
        
        # Qiskit Benchmark
        qiskit_params = np.array(initial_params, dtype=np.float64) 
        q_time, q_mem = benchmark_qiskit(NUM_QUBITS, reps, qiskit_params)
        
        # PennyLane Benchmark
        pl_params = pnp.array(initial_params, requires_grad=True)
        pl_time, pl_mem = benchmark_pennylane(NUM_QUBITS, reps, pl_params)
        
        results.append({
            "num_qubits": NUM_QUBITS,
            "reps": reps,
            "num_params": num_params,
            "qiskit_time_s": q_time,
            "qiskit_memory_mb": q_mem,
            "pennylane_time_s": pl_time,
            "pennylane_memory_mb": pl_mem,
        })

        print(f"{num_params:<12} | {q_time:<12.4f} s | {q_mem:<10.2f} MB | {pl_time:<15.4f} s | {pl_mem:<10.2f} MB")

    with open(RESULTS_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved benchmark results to {RESULTS_FILE}")