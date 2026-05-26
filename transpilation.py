import csv
import time
import json
import tracemalloc
import pennylane as qml
from pennylane_qiskit.converter import circuit_to_qiskit
from qiskit.circuit.library import EfficientSU2
from qiskit import transpile
from qiskit.transpiler import CouplingMap

def benchmark_deployment_pipeline(num_qubits, reps, topology_type):
    # Define the physical hardware constraints
    if topology_type == "IBM_Line":
        cmap = CouplingMap.from_line(num_qubits)
    elif topology_type == "Rigetti_Ring":
        cmap = CouplingMap.from_ring(num_qubits)
    elif topology_type == "IonQ_AlltoAll":
        cmap = CouplingMap.from_full(num_qubits)

    # ==========================================
    # Native Qiskit Benchmark
    # ==========================================
    tracemalloc.start()
    q_start = time.perf_counter()
    
    # Build Logical
    q_logical = EfficientSU2(num_qubits, reps=reps, entanglement='linear').decompose()
    
    # METRIC: Logical Gate Profile
    q_logical_profile = dict(q_logical.count_ops())
    q_logical_cx = q_logical_profile.get('cx', 0)
    
    # Transpile
    q_physical = transpile(q_logical, coupling_map=cmap, optimization_level=3, basis_gates=['rz', 'sx', 'x', 'cx'], seed_transpiler=42)
    
    q_end = time.perf_counter()
    
    # METRIC: Peak Memory
    _, q_peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    q_peak_mem_mb = q_peak_mem_bytes / (1024 * 1024)
    
    q_time = q_end - q_start
    q_physical_cx = q_physical.count_ops().get('cx', 0)
    q_added_cx = q_physical_cx - q_logical_cx

    # ==========================================
    # PennyLane Cross-Compilation Benchmark
    # ==========================================
    tracemalloc.start()
    pl_start = time.perf_counter()
    
    # Build Logical QNode
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def pl_circuit():
        for i in range(num_qubits):
            qml.RY(0.1, wires=i)
            qml.RZ(0.1, wires=i)
        for _ in range(reps):
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            for i in range(num_qubits):
                qml.RY(0.1, wires=i)
                qml.RZ(0.1, wires=i)
        return qml.state()
    
    # Use official Xanadu translation bridge
    pl_tapes, _ = qml.workflow.construct_batch(pl_circuit)()
    pl_translated = circuit_to_qiskit(pl_tapes[0], register_size=num_qubits, measure=False)
    
    # METRIC: Logical Gate Profile (After translation, before hardware routing)
    pl_logical_profile = dict(pl_translated.count_ops())
    
    # Transpile (Route to hardware using the translated circuit)
    pl_physical = transpile(pl_translated, coupling_map=cmap, optimization_level=3, basis_gates=['rz', 'sx', 'x', 'cx'], seed_transpiler=42)
    
    pl_end = time.perf_counter()
    
    # METRIC: Peak Memory
    _, pl_peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pl_peak_mem_mb = pl_peak_mem_bytes / (1024 * 1024)
    
    pl_time = pl_end - pl_start
    
    # Verify the physical outputs are identical
    pl_physical_cx = pl_physical.count_ops().get('cx', 0)
    pl_added_cx = pl_physical_cx - q_logical_cx

    return q_time, pl_time, q_peak_mem_mb, pl_peak_mem_mb, q_added_cx, pl_added_cx, q_logical_profile, pl_logical_profile

if __name__ == "__main__":
    FIXED_REPS = 2  
    QUBIT_SCALES = [4, 8, 16, 32, 64, 128, 256]
    TOPOLOGIES = ["IBM_Line", "Rigetti_Ring", "IonQ_AlltoAll"]
    RESULTS_FILE = "transpilation_benchmark_results.csv"
    
    # Updated fieldnames for CSV
    fieldnames = [
        "topology",
        "num_qubits",
        "reps",
        "qiskit_time_s",
        "pennylane_time_s",
        "qiskit_peak_mem_mb",
        "pennylane_peak_mem_mb",
        "added_swaps_control", # Combined since they are identical
        "qiskit_logical_profile",
        "pennylane_logical_profile"
    ]
    
    with open(RESULTS_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        print("--- Phase 2: Hardware Deployment Benchmark ---")
        for topology in TOPOLOGIES:
            print(f"\nTarget Architecture: {topology}")
            print(f"{'Qubits':<8} | {'Qiskit Time':<12} | {'PL Time':<12} | {'Qiskit RAM(MB)':<15} | {'PL RAM(MB)':<15} | {'Added SWAPs'}")
            print("-" * 88)
            
            for n in QUBIT_SCALES:
                (q_time, pl_time, q_mem, pl_mem, q_swaps, pl_swaps, 
                 q_prof, pl_prof) = benchmark_deployment_pipeline(n, FIXED_REPS, topology)
                
                # Control Variable check
                assert q_swaps == pl_swaps, "Mismatch in physical circuits, test invalid."

                writer.writerow({
                    "topology": topology,
                    "num_qubits": n,
                    "reps": FIXED_REPS,
                    "qiskit_time_s": q_time,
                    "pennylane_time_s": pl_time,
                    "qiskit_peak_mem_mb": q_mem,
                    "pennylane_peak_mem_mb": pl_mem,
                    "added_swaps_control": q_swaps,
                    "qiskit_logical_profile": json.dumps(q_prof),
                    "pennylane_logical_profile": json.dumps(pl_prof),
                })
                file.flush()
                
                print(f"{n:<8} | {q_time:<12.4f} | {pl_time:<12.4f} | {q_mem:<15.2f} | {pl_mem:<15.2f} | {q_swaps}")

    print(f"\nSaved benchmark results to {RESULTS_FILE}")