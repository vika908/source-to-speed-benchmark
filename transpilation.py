import csv
import time
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

    # Native Qiskit Benchmark
    q_start = time.perf_counter()
    
    # Build Logical
    q_logical = EfficientSU2(num_qubits, reps=reps, entanglement='linear').decompose()
    q_logical_cx = q_logical.count_ops().get('cx', 0)
    
    # Transpile
    q_physical = transpile(q_logical, coupling_map=cmap, optimization_level=3, basis_gates=['rz', 'sx', 'x', 'cx'], seed_transpiler=42)
    
    q_end = time.perf_counter()
    q_time = q_end - q_start
    q_physical_cx = q_physical.count_ops().get('cx', 0)
    q_added_cx = q_physical_cx - q_logical_cx

    # PennyLane Cross-Compilation Benchmark
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
    
    # Transpile (Route to hardware using the translated circuit)
    pl_physical = transpile(pl_translated, coupling_map=cmap, optimization_level=3, basis_gates=['rz', 'sx', 'x', 'cx'], seed_transpiler=42)
    
    pl_end = time.perf_counter()
    pl_time = pl_end - pl_start
    
    # Verify the physical outputs are identical
    pl_physical_cx = pl_physical.count_ops().get('cx', 0)
    pl_added_cx = pl_physical_cx - q_logical_cx

    return q_time, pl_time, q_added_cx, pl_added_cx

if __name__ == "__main__":
    FIXED_REPS = 2  
    QUBIT_SCALES = [4, 8, 16, 32, 64, 128, 256]
    TOPOLOGIES = ["IBM_Line", "Rigetti_Ring", "IonQ_AlltoAll"]
    RESULTS_FILE = "transpilation_benchmark_results.csv"
    fieldnames = [
        "topology",
        "num_qubits",
        "reps",
        "qiskit_time_s",
        "pennylane_time_s",
        "qiskit_added_swaps_control",
        "pennylane_added_swaps_control",
    ]
    
    with open(RESULTS_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for topology in TOPOLOGIES:
            for n in QUBIT_SCALES:
                q_time, pl_time, q_swaps, pl_swaps = benchmark_deployment_pipeline(n, FIXED_REPS, topology)
                
                # Verify the physical outputs are identical
                assert q_swaps == pl_swaps, "Mismatch in physical circuits, test invalid."

                writer.writerow({
                    "topology": topology,
                    "num_qubits": n,
                    "reps": FIXED_REPS,
                    "qiskit_time_s": q_time,
                    "pennylane_time_s": pl_time,
                    "qiskit_added_swaps_control": q_swaps,
                    "pennylane_added_swaps_control": pl_swaps,
                })
                file.flush()
                
                print(f"{n:<12} | {q_time:<15.4f} | {pl_time:<18.4f} | {q_swaps}")

    print(f"\nSaved benchmark results to {RESULTS_FILE}")