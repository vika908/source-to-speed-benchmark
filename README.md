# Benchmark of Qiskit vs Pennylane with Quantum Machine Learning algorithms
The two frameworks possess fundamentally different standard gradient estimation methods. Qiskit's native method for estimating the gradient relies on the parameter-shift rule, while PennyLane relies heavily on adjoint differentiation. These two methods calculate the derivative of the cost function in fundamentally different ways: the parameter-shift rule scales linearly $O(M)$ with respect to the number of parameters, whereas adjoint differentiation scaling remains $O(1)$. Therefore, the most critical architectural differences between these two frameworks are exposed in algorithms with massive parameter counts. (More details in the LaTeX paper).   Considering this, I chose to benchmark these frameworks using Quantum Machine Learning (QML). QML is a family of algorithms designed for model training and is part of the broader group of Variational Quantum Algorithms (VQAs). The primary reason for selecting this family is that QML architectures require thousands of parameters to achieve sufficient data expressivity, making it perfect for demonstrating the scaling differences between Qiskit and PennyLane.  
In reality, a trained model is of no practical use if it remains as a logical circuit. One must deploy the model onto physical hardware to utilize it.

## Gradient Phase 
As mentioned above, one of the differences in Qiskit and Pennylane architectures is calculation of the gradient. Isolating this phase from the cost function calculation, classical optimizer and data loading makes the result clearer as it highlights the exact mathematical dissimilarity of the two discussed frameworks.  
Measured metrics:  
* Time to estimate the gradient  
* Peak memory footprint 

## Transpilation Phase
Pennylane is software-oriented framework and it does not possess any native circuit compiler, instead it uses other third-party frameworks, such as Qiskit itself, to perform physical transpilation.  
Measured metrics:  
* Time to transpile  
* Peak memeory footprint
* Number of inserted SWAP gates (Control variable)
* Circuit's depth (Control variable)
* Logical basis gate dictionary 