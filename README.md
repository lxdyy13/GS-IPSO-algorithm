# Path Planning Demo

This project demonstrates an improved Particle Swarm Optimization algorithm for 3D path planning. It includes:

- Reading and parsing city/coordinate data.
- Implementing an improved PSO algorithm (GSIPSO).
- Executing the path search and outputting optimal results.
- Visualizing the path and convergence curves.

---

## Environment and Dependencies

### Python Version

- Python >= 3.8

### Required Libraries

Install required libraries via `pip`:  pip3 install numpy matplotlib


## How to use

### Prepare the Data

In the same directory as GS-IPSO.py, create (or use) view_point.txt

### Run the Project

Open your terminal/command prompt in the project directory. python3 GS-IPSO.py

### Check Results

Console Output:

Shows the integrated cost (distance + penalties) each iteration.

Prints final best values (path length, penalty counts, etc.).

Generated Files:

Best_path.txt: Coordinates of the best path.

iteration_data.txt: Iteration index and global best values.

Plots:

3D route path.

Convergence curve.
