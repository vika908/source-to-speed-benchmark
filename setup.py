from setuptools import setup, find_packages

setup(
    name="vqml_benchmarks",
    version="0.1.0",
    description="Benchmarking Qiskit vs PennyLane for VQML Training and Deployment",
    author="Viktoriia Hrodetska",
    packages=find_packages(),
    install_requires=[
        # Qiskit
        "qiskit>=1.0.0",           # Qiskit library
        "qiskit-aer",              # Qiskit simulator
        "qiskit-ibm-runtime",      # Qiskit IBM runtime
        "qiskit-algorithms",      # Qiskit algorithms
        
        # PennyLane
        "pennylane",               # PennyLane library
        "pennylane-lightning",     # PennyLane lightning backend
        "pennylane-qiskit",        # PennyLane Qiskit plugin
        "pennylane-catalyst",      # PennyLane Catalyst JIT compiler
        "autoray<0.8.0",
        
        # Math Utilities
        "numpy",
        "matplotlib",              # Matplotlib for plotting
        "memory-profiler",         # Memory profiler for tracking RAM usage
        "psutil"                   # System utilities for precise timing/memory tracking
    ],
    python_requires=">=3.9",
)