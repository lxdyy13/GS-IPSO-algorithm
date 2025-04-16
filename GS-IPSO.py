import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------------------
#           Helper Functions
# ----------------------------------------

def read_coordinates(data_string):
    """
    Reads 3D coordinates from a string. Each valid line in 'data_string'
    should contain exactly three floating-point numbers.

    Args:
        data_string (str): A multi-line string containing coordinate data,
                           where each line has 'x y z'.

    Returns:
        List[List[float]]: A list of [x, y, z] coordinates.
    """
    coordinates = []
    lines = data_string.strip().split('\n')
    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            coordinates.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return coordinates

def save_to_txt(iter_x, iter_y, filename='iteration_data.txt'):
    """
    Saves iteration data (iteration index and corresponding global best value)
    to a text file.

    Args:
        iter_x (List[int]): A list of iteration indices.
        iter_y (List[float]): A list of global best values at each iteration.
        filename (str, optional): The file name for saving the iteration data.
                                  Defaults to 'iteration_data.txt'.
    """
    with open(filename, 'w') as f:
        for x, y in zip(iter_x, iter_y):
            f.write(f"{x} {y}\n")

def plot_best_path(best_path):
    """
    Plots the best path in 3D space, illustrating the route taken by the solution.

    Args:
        best_path (np.ndarray): An array of shape (N, 3) representing the best route.
                                Each row contains [x, y, z] coordinates of a point.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_coords = best_path[:, 0]
    y_coords = best_path[:, 1]
    z_coords = best_path[:, 2]
    ax.scatter(x_coords, y_coords, z_coords, c='k', marker='o')
    ax.plot(x_coords, y_coords, z_coords, c='r')
    ax.set_title('PSO')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_convergence(iter_x, iter_y):
    """
    Plots the convergence curve showing the global best value versus iteration.

    Args:
        iter_x (List[int]): Iteration indices.
        iter_y (List[float]): Corresponding global best values (fitness or cost).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iter_x, iter_y, linestyle='-', color='g', label='01 Length')
    plt.xlabel('Iteration')
    plt.ylabel('Global Best Length')
    plt.title('Convergence of PSO Algorithm')
    plt.grid(True)
    plt.legend()
    plt.show()

def line_box_intersect(p1, p2):
    """
    Checks if the line segment from p1 to p2 intersects with a predefined rectangular box.
    If necessary, add or delete.
    The box is fixed in the code with box_min = [0, 0, 0] and box_max = [30, 30, 30].

    Args:
        p1 (np.ndarray): Start point of the segment, shape=(3,).
        p2 (np.ndarray): End point of the segment, shape=(3,).

    Returns:
        bool: True if the line segment intersects with the box, False otherwise.
    """
    box_min = np.array([0, 0, 0])
    box_max = np.array([5000, 5000, 5000])
    
    direction = p2 - p1   # Vector from p1 to p2
    tmin, tmax = 0.0, 1.0
    
    for i in range(3):
        # If there's no movement in the i-th dimension:
        if abs(direction[i]) < 1e-12:
            # If p1 is outside the box range in this dimension, then no intersection is possible
            if p1[i] < box_min[i] or p1[i] > box_max[i]:
                return False
        else:
            # Calculate intersection parameters with the two planes in the i-th dimension
            ood = 1.0 / direction[i]
            t1 = (box_min[i] - p1[i]) * ood
            t2 = (box_max[i] - p1[i]) * ood
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            # If the intersection parameter range is invalid, there's no intersection
            if tmin > tmax:
                return False

    # If [tmin, tmax] overlaps with the segment parameter [0,1], it intersects the box
    return (tmax >= 0.0) and (tmin <= 1.0)

# ----------------------------------------
#           GSIPSO Class Definition
# ----------------------------------------

class GSIPSO(object):
    """
    A class that implements a hybrid PSO (Particle Swarm Optimization) with
    genetic algorithm-like crossover and mutation operators, plus a simulated
    annealing-based acceptance criterion.
    """
    def __init__(self, num_city, data):
        """
        Initializes the GSIPSO object with given data and parameters.

        Args:
            num_city (int): Number of points or cities.
            data (np.ndarray): Coordinate data of shape (num_city, 3).
        """
        # Constraint parameters for sharp turns and rapid ascent
        self.angle_size = 1       # Threshold for sharp turns, angle < pi/self.angle_size triggers penalty
        self.high_size = 5000     # Threshold for rapid ascent, height difference triggers penalty

        self.iter_max = 300       # Maximum number of iterations
        self.num = 800            # Number of particles in the swarm
        self.num_city = num_city  # Number of cities or points
        self.location = data      # Storing coordinates for all points

        # Compute the distance matrix for all city pairs
        self.dis_mat = self.compute_dis_mat(num_city, self.location)

        # Initialize particles using a greedy strategy
        self.particals = self.greedy_init(self.dis_mat, self.num, num_city)

        # Calculate path costs for the initial set of particles
        self.lenths = self.compute_paths(self.particals)

        # Identify the best initial solution among the particles
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]

        # Set up local bests and global bests
        self.local_best = self.particals.copy()
        self.local_best_len = self.lenths.copy()
        self.global_best = init_path
        self.global_best_len = init_l

        # Track the best length and path
        self.best_l = self.global_best_len
        self.best_path = self.global_best

        # Variables for plotting convergence
        self.iter_x = [0]
        self.iter_y = [init_l]

        # selection parameters
        self.num_total = self.num
        self.elite_choose_ratio = 0.4
        self.elite_mutate_ratio = 0.05
        self.fruits = self.particals.copy()

        # A partition of the iteration process into early stage and later stage
        self.T_early = int(0.4 * self.iter_max)

        # Initial temperature for simulated annealing acceptance probability
        self.init_temp = self.global_best_len

    def greedy_init(self, dis_mat, num_total, num_city):
        """
        Generates an initial population of solutions (particles) using a greedy method.
        Each solution is built by always moving to the nearest unvisited city.

        Args:
            dis_mat (np.ndarray): Distance matrix of size (num_city, num_city).
            num_total (int): Number of particles to create.
            num_city (int): Number of cities or points.

        Returns:
            List[List[int]]: A list of path permutations, each path is a list of city indices.
        """
        result = []
        for i in range(num_total):
            rest = list(range(num_city))
            current = i % num_city
            rest.remove(current)
            path = [current]
            while rest:
                next_city = min(rest, key=lambda x: dis_mat[current][x])
                path.append(next_city)
                rest.remove(next_city)
                current = next_city
            result.append(path)
        return result
    
    def metropolis_accept_prob(self, J_current, J_new):
        """
        Computes the simulated annealing acceptance probability (Metropolis criterion).

        Args:
            J_current (float): Current cost or fitness value.
            J_new (float): New proposed cost or fitness value.

        Returns:
            float: Probability of accepting the new solution, even if worse.
        """
        return np.exp((J_current - J_new) / self.init_temp)  
    
    def adaptive_alpha(self, t):
        """
        Computes adaptive crossover probabilities alpha1 and alpha2.

        - During the early stage (first T_early iterations), alpha1 decreases linearly from 1.0 to 0,
          while alpha2 remains 0.
        - In the later stage, alpha1 remains 0, alpha2 decreases linearly from 1.0 to 0.

        Args:
            t (int): Current iteration index.

        Returns:
            (float, float): Tuple (alpha1, alpha2).
        """
        if t <= self.T_early:
            alpha1 = 1 - t / self.T_early
            alpha2 = 0
        else:
            alpha1 = 0
            alpha2 = 1 - (t - self.T_early) / (self.iter_max - self.T_early)
        return alpha1, alpha2

    def compute_dis_mat(self, num_city, location):
        """
        Computes pairwise Euclidean distances between all cities/points.

        Args:
            num_city (int): Number of cities/points.
            location (np.ndarray): Coordinates of shape (num_city, 3).

        Returns:
            np.ndarray: A 2D array (num_city x num_city) of pairwise distances.
        """
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                else:
                    dis_mat[i][j] = np.linalg.norm(location[i] - location[j])
        return dis_mat

    def compute_pathlen(self, path, dis_mat):
        """
        Computes the total cost (distance + penalties) of a given path.
        The path cost includes:
          - Basic traveling distance
          - Sharp turn penalty
          - Rapid ascent penalty
          - Collision penalty (if the segment intersects the safety box [0,0,0] to [30,30,30])

        Args:
            path (List[int]): A sequence of city indices representing the route.
            dis_mat (np.ndarray): Precomputed distance matrix.

        Returns:
            float: The computed path cost.
        """
        loc = self.location[path]
        # Compute total traveling distance, including returning to the start city
        dist = np.sum(dis_mat[path[:-1], path[1:]]) + dis_mat[path[-1]][path[0]]

        # Sharp turn penalty
        vectors = loc[1:] - loc[:-1]
        norms = np.linalg.norm(vectors, axis=1)
        unit_vectors = vectors / norms[:, None]
        cos_thetas = np.einsum('ij,ij->i', unit_vectors[:-1], unit_vectors[1:])
        thetas = np.arccos(np.clip(cos_thetas, -1.0, 1.0))
        angle_penalty = np.sum(thetas < (math.pi / self.angle_size))

        # Rapid ascent penalty
        height_diff = np.abs(loc[1:, 2] - loc[:-1, 2])
        height_penalty = np.sum(height_diff > self.high_size)

        # Collision penalty: If any segment intersects the safety box, add a large penalty
        collision_penalty = 0
        for i in range(len(path) - 1):
            p1 = loc[i]
            p2 = loc[i+1]
            if line_box_intersect(p1, p2):
                collision_penalty += 1e8
                break  # Once collision is found, we can stop checking

        return dist / 1000 + angle_penalty + height_penalty + collision_penalty

    def compute_paths(self, paths):
        """
        Computes the cost of each path in a list of paths.

        Args:
            paths (List[List[int]]): A list of path permutations.

        Returns:
            List[float]: A list of path costs, one for each path.
        """
        return [self.compute_pathlen(path, self.dis_mat) for path in paths]

    def eval_particals(self):
        """
        Evaluates the current swarm and updates global and local best solutions.
        """
        min_l = min(self.lenths)
        min_index = self.lenths.index(min_l)
        # If a new global best is found, update it
        if min_l < self.global_best_len:
            self.global_best_len = min_l
            self.global_best = self.particals[min_index]

        # Update local best for each particle
        for i in range(len(self.lenths)):
            if self.lenths[i] < self.local_best_len[i]:
                self.local_best_len[i] = self.lenths[i]
                self.local_best[i] = self.particals[i]

    def cross(self, cur, best):
        """
        Crosses over two routes. Selects a slice of 'best' and merges it into 'cur',
        creating two candidate solutions, and returns the better one.

        Args:
            cur (List[int]): The current path.
            best (List[int]): The best path to guide crossover.

        Returns:
            (List[int], float): A tuple containing the better new path and its cost.
        """
        l = sorted(random.sample(range(self.num_city), 2))
        cross_part = best[l[0]:l[1]]
        tmp = [x for x in cur if x not in cross_part]
        candidate1 = tmp + cross_part
        candidate2 = cross_part + tmp
        cost1 = self.compute_pathlen(candidate1, self.dis_mat)
        cost2 = self.compute_pathlen(candidate2, self.dis_mat)
        return (candidate1, cost1) if cost1 < cost2 else (candidate2, cost2)

    def mutate(self, one):
        """
        A simple mutation operator that randomly swaps two positions in the path.

        Args:
            one (List[int]): The path to be mutated.

        Returns:
            (List[int], float): The mutated path and its new cost.
        """
        one = one.copy()
        x, y = sorted(random.sample(range(self.num_city), 2))
        one[x], one[y] = one[y], one[x]
        return one, self.compute_pathlen(one, self.dis_mat)

    def compute_adp(self, fruits):
        """
        Calculates the fitness (1 / cost) for a list of paths.

        Args:
            fruits (List[List[int]]): A list of paths (city index permutations).

        Returns:
            np.ndarray: An array of fitness values for each path.
        """
        return np.array([1.0 / self.compute_pathlen(f, self.dis_mat) for f in fruits])

    def elite_parent(self, scores, ratio):
        """
        Selects the top (ratio * total) solutions by their fitness (descending order).

        Args:
            scores (np.ndarray): An array of fitness values corresponding to 'self.fruits'.
            ratio (float): The fraction of top individuals to be selected.

        Returns:
            (List[List[int]], List[float]): A tuple of (selected_paths, selected_scores).
        """
        idx = np.argsort(-scores)[:int(ratio * len(scores))]
        return [self.fruits[i] for i in idx], [scores[i] for i in idx]

    def elite_cross(self, x, y):
        """
        A crossover function that takes a slice of X and swaps it with Y.

        Args:
            x (List[int]): Parent path X.
            y (List[int]): Parent path Y.

        Returns:
            (List[int], List[int]): Two offspring after crossover.
        """
        start, end = sorted(random.sample(range(len(x)), 2))
        tmp_x, tmp_y = x[start:end], y[start:end]
        x_new = y.copy()
        y_new = x.copy()
        for i in range(start, end):
            x_new[i], y_new[i] = y[i], x[i]
        return x_new, y_new

    def elite_mutate(self, gene):
        """
        Mutation operator that reverses a slice of the path.

        Args:
            gene (List[int]): A path to mutate.

        Returns:
            List[int]: The mutated path.
        """
        start, end = sorted(random.sample(range(len(gene)), 2))
        gene[start:end] = gene[start:end][::-1]
        return gene

    def elite_choose(self, scores, parents):
        """
        Chooses two parents using roulette-wheel selection (based on normalized fitness).

        Args:
            scores (List[float]): Fitness values of the parent group.
            parents (List[List[int]]): The parent paths.

        Returns:
            (List[int], List[int]): Two chosen parents for crossover.
        """
        probs = np.array(scores) / sum(scores)
        idx1, idx2 = np.random.choice(len(parents), 2, p=probs)
        return parents[idx1].copy(), parents[idx2].copy()

    def roulette_selection_function(self):
        """
        Roulette_selection that uses elite selection and roulette-wheel to generate offspring.
        This function:
          1. Selects elite parents.
          2. Performs crossover.
          3. Performs mutation.
          4. Rebuilds 'self.fruits' and 'self.particals' with new solutions.

        Returns:
            (List[int], float): The best path among the chosen parents and its fitness.
        """
        scores = self.compute_adp(self.fruits)
        parents, parent_scores = self.elite_parent(scores, self.elite_choose_ratio)
        best_gene = parents[0]
        best_score = parent_scores[0]
        fruits = parents.copy()

        # Fill population up to self.num_total
        while len(fruits) < self.num_total:
            g1, g2 = self.elite_choose(parent_scores, parents)
            c1, c2 = self.elite_cross(g1, g2)
            # Mutate with a certain probability
            if random.random() < self.elite_mutate_ratio:
                c1 = self.elite_mutate(c1)
            if random.random() < self.elite_mutate_ratio:
                c2 = self.elite_mutate(c2)
            for gene in [c1, c2]:
                if gene not in fruits:
                    fruits.append(gene)
                    if len(fruits) >= self.num_total:
                        break
        self.fruits = fruits
        self.particals = self.fruits
        return best_gene, best_score

    def print_compute_penalties(self, path, dis_mat):
        """
        Analyzes a given path to separately compute:
          - The sharp turn count
          - The rapid ascent count
          - The raw distance (in kilometers), excluding collision penalty

        This function is mainly for reporting or debugging purposes.

        Args:
            path (List[int]): A permutation of city indices representing the route.
            dis_mat (np.ndarray): The distance matrix for all cities.

        Returns:
            (int, int, float): A tuple of:
                                (number_of_sharp_turns, number_of_rapid_ascents, distance_in_km).
        """
        loc = self.location[path]
        dist = np.sum(dis_mat[path[:-1], path[1:]]) + dis_mat[path[-1]][path[0]]

        vectors = loc[1:] - loc[:-1]
        norms = np.linalg.norm(vectors, axis=1)
        unit_vectors = vectors / norms[:, None]
        cos_thetas = np.einsum('ij,ij->i', unit_vectors[:-1], unit_vectors[1:])
        thetas = np.arccos(np.clip(cos_thetas, -1.0, 1.0))
        angle_penalty = np.sum(thetas < (math.pi / self.angle_size))

        height_diff = np.abs(loc[1:, 2] - loc[:-1, 2])
        height_penalty = np.sum(height_diff > self.high_size)

        return angle_penalty, height_penalty, dist / 1000

    def gsipso(self):
        """
        Main iteration of the GS-IPSO approach, which integrates:
          1. A roulette-like step (roulette_selection_function) to foster new solutions.
          2. Particle update with cross and mutate operators influenced by alpha1, alpha2,
             local best, and global best solutions.
          3. Metropolis acceptance for letting worse solutions in to escape local minima.

        Returns:
            (float, List[int]): (global_best_cost, global_best_path).
        """
        for cnt in range(1, self.iter_max):
            # Determine adaptive factors for this iteration
            alpha1, alpha2 = self.adaptive_alpha(cnt)

            # Perform selection to generate improved solutions
            self.best_path, self.best_l = self.roulette_selection_function()
            # Convert the best solution's fitness to a cost
            self.best_l = 1. / self.best_l

            # Recompute the cost for all current particles (paths)
            self.lenths = self.compute_paths(self.particals)

            # Update each particle based on alpha1, alpha2, etc.
            for i, one in enumerate(self.particals):
                # 1) Crossover with a random particle (controlled by alpha1)
                if random.random() < alpha1:
                    rand_idx = random.randint(0, self.num - 1)
                    new_one, new_l = self.cross(one, self.particals[rand_idx])
                    if new_l < self.lenths[i]:
                        # Accept with a Metropolis probability
                        if random.random() < self.metropolis_accept_prob(self.lenths[i], new_l):
                            self.particals[i] = new_one
                            self.lenths[i] = new_l

                # 2) Crossover with the local best (controlled by alpha2)
                if random.random() < alpha2:
                    new_one, new_l = self.cross(one, self.local_best[i])
                    if new_l < self.lenths[i]:
                        if random.random() < self.metropolis_accept_prob(self.lenths[i], new_l):
                            self.particals[i] = new_one
                            self.lenths[i] = new_l

                # 3) Crossover with the global best (10% probability)
                if random.random() < 0.1:
                    new_one, new_l = self.cross(self.particals[i], self.global_best)
                    if new_l < self.lenths[i]:
                        if random.random() < self.metropolis_accept_prob(self.lenths[i], new_l):
                            self.particals[i] = new_one
                            self.lenths[i] = new_l

                # 4) Mutation with 20% probability
                if random.random() < 0.2:
                    new_one, new_l = self.mutate(self.particals[i])
                    if new_l < self.lenths[i]:
                        if random.random() < self.metropolis_accept_prob(self.lenths[i], new_l):
                            self.particals[i] = new_one
                            self.lenths[i] = new_l

            # After updating, evaluate the swarm to refresh local/global best solutions
            self.eval_particals()

            # If the best candidate is better than our global best, update global best
            if self.best_l < self.global_best_len:
                self.global_best_len = self.best_l
                self.global_best = self.best_path

            # Print iteration results
            print(f"Iteration {cnt}: Current Best Cost = {self.global_best_len:.2f}")

            # Record convergence info
            self.iter_x.append(cnt)
            self.iter_y.append(self.global_best_len)

        return self.global_best_len, self.global_best

    def run(self):
        """
        Runs the entire gsipso process and prints final details like
        total distance, number of rapid ascents, and number of sharp turns.

        Returns:
            (np.ndarray, float): A tuple of (final_best_route_coordinates, final_best_cost).
        """
        best_length, best_path = self.gsipso()
        final_angle_penalty, final_height_penalty, final_distance = self.print_compute_penalties(best_path, self.dis_mat)
        print(f"\nFinal Path Distance (km): {final_distance}")
        print(f"Number of Rapid Ascents: {final_height_penalty}")
        print(f"Number of Sharp Turns: {final_angle_penalty}")
        return self.location[best_path], best_length

# ----------------------------------------
#           Main Program Entry
# ----------------------------------------

def main():
    """
    Main entry point:
    1. Reads the coordinate file 'view_point.txt' to obtain 3D points.
    2. Creates a GSIPSO instance.
    3. Runs the optimization and measures runtime.
    4. Outputs the best route, convergence, and relevant metrics.
    """
    file_path = "view_point.txt"
    with open(file_path, 'r') as f:
        data_string = f.read()
    data = np.array(read_coordinates(data_string))
    num_coordinates = len(data)

    # Instantiate the GSIPSO model
    model = GSIPSO(num_city=num_coordinates, data=data.copy())

    start_time = time.time()
    best_path, best_value = model.run()
    end_time = time.time()

    print(f"\nGlobal Best Fitness Value: {best_value}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")

    # Save and visualize the best route
    extended_best_path = np.vstack([best_path, best_path[0]])
    np.savetxt('Best_path.txt', extended_best_path)
    plot_best_path(extended_best_path)
    plot_convergence(model.iter_x, model.iter_y)
    save_to_txt(model.iter_x, model.iter_y)

if __name__ == "__main__":
    main()
