# Unconstrained Derivative-based Optimization

**Politecnico di Torino** *Course: Numerical Optimization for Large Scale Problems (A.Y. 2025/2026)*

## Project Description
This project implements and analyzes algorithms for unconstrained derivative-based numerical optimization, focusing specifically on solving large-scale problems. The software evaluates the mathematical performance of various methods and approximation techniques, comparing key metrics such as the number of iterations, execution times, and success rates.

## Implemented Algorithms
The algorithmic core of the software focuses on the following methodologies:

* **Optimization Methods:**
    * **Modified Newton Method:** Optimization via modified Cholesky factorization to guarantee safe descent directions even with non-positive definite matrices.
    * **Truncated Newton Method (Inexact Newton):** Approximate resolution of the linear system to scale efficiently on large problems.
* **Finite Differences Approximation:**
    * Gradient and Jacobian approximation (focusing on diagonals for memory efficiency).
    * Hessian approximation (pentadiagonal structure) computed via finite differences of the gradient.
    * Dynamic management and stability study as the *step size* ($h$) varies.
* **Test Models (Benchmarks):**
    * **Problem 31:** Broyden problem.
    * **Problem 52:** Trigonometric exponential system.

## Results and Visualizations
Below are the summary performance charts for the analyzed benchmarks. The plots illustrate the behavior of the algorithms applied to the mathematical test models.

### Broyden Problem (Problem 31)
Algorithm performance applied to the Broyden problem:
![Plot P31 Exact Truncated Newton](figures/plots/p31_plots/Plot_P31_exact_tr.png)

### Trigonometric Exponential System (Problem 52)
Algorithm performance applied to the Trigonometric exponential system:
![Plot P52 Exact Truncated Newton](figures/plots/p52_plots/Plot_P52_exact_tr.png)

---

## Authors
* **Chaoui Aziz Riym** (352283)
* **Egidio Elio** (359806)
* **Paoletti Simone** (359956)