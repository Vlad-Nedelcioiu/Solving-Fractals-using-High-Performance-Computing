
# Solving Fractals using High Performance Computing

## Overview
This project explores the implementation of fractal solving algorithms (e.g., Julia sets) using various high-performance computing techniques.  
It serves as a study case for parallel and distributed computation applied to computationally intensive graphics tasks.

---

## Features
- C and C++ implementations of fractal computations
- OpenMP parallelization for shared memory systems
- MPI (Message Passing Interface) versions for distributed systems
- Logs and summaries of performance tests
- Example virtual environment setup for supporting Python scripts

---

## Repository Structure
```
Solving-Fractals-using-High-Performance-Computing/
├── app-labs/
│   └── julia/
│       ├── julia_openmp.c
│       ├── julia_hybrid.c
│       ├── logs/
│       ├── summaries/
│       └── venv/                  # virtual environment (excluded from Git tracking)
├── .vscode/                        # VSCode settings (optional)
├── .gitignore                      # ignored files and folders
└── README.md                       # this file
```

---

## How to Build and Run

### 1. Build OpenMP version
```bash
gcc -fopenmp julia/julia_openmp.c -o julia_openmp
```

### 2. Build hybrid (OpenMP + MPI) version
```bash
mpicc -fopenmp julia/julia_hybrid.c -o julia_hybrid
```

### 3. Run examples
```bash
./julia_openmp
mpirun -np <num_procs> ./julia_hybrid
```

Replace `<num_procs>` with the desired number of MPI processes.

---

## Dependencies
- GCC (with OpenMP support)
- MPI (e.g., OpenMPI or MPICH)
- Python (for optional log processing / scripting)
- Make (optional, if you create a Makefile)

---

## Notes
- Virtual environments (venv/) and generated binaries are excluded from version control.
- Logs and summaries demonstrate timing and performance of various scheduling strategies.

---

## License
This project is released under the MIT License.

---

## Author
**Vlad Nedelcioiu**  
For suggestions, improvements or questions, feel free to open an issue or pull request.

