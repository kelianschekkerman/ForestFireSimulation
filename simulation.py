import numpy as np
import noise
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation

# Take the arguments from the command line and return them
def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distribution", default='random', type=str,
                        help="Distribution of tree placement. Options are: random, semi-random, ordered")
    parser.add_argument("-t", "--treeprobability", default=0.6, type=float,
                        help="Probability of a tree being placed.")
    parser.add_argument("-f", "--fireprobability", default=0.6, type=float,
                        help="Probability of trees catching fire.")
    parser.add_argument("-s", "--size", default=100, type=int,
                        help="The size of the square lattice.")
    args = parser.parse_args() 
    return args

# Initialize the square lattice with trees based on the distribution type
def init_square_lattice(n, p, type):
    # random tree distribution on the lattice
    if type == 'random': 
        lattice = np.random.choice([0, 1], size=(n, n), p=[1-p, p])

    # TODO: implement variety with p on the ordered lattice
    # (possibly add scatter so the trees can jump between lines)
    # vertical tree lines on the lattice
    elif type == 'ordered': 
        lattice = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j % 2 == 0:
                    lattice[i, j] = 1

    # TODO: implement variety with p on the semi-ordered lattice
    # random tree distribution based on perlin noise
    elif type == 'semi-random': 
        lattice = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                result = noise.pnoise2(i/n, j/n, octaves=10, persistence=1, lacunarity=20.0)
                lattice[i, j] = 0 if result < 0 else 1

    return lattice

# Return a single image used in a matplotlib animation
def return_image(lattice, title="Forest fire"):
    cmap = ListedColormap(['w', 'g', 'r', 'k'])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    # plt.title(title) # TODO: this crashes the animation for some reason
    # plt.axis('off')
    im = plt.imshow(lattice, cmap=cmap, norm=norm, animated=True)
    return [im]

# Start the fire by setting a random tree on fire
def random_arson(grid):
    indices = np.argwhere(grid == 1)
    random_index = np.random.choice(indices.shape[0])
    random_point = indices[random_index]
    grid[random_point[0], random_point[1]] = 2

# Return true if site at i,j is inside of the lattice for the edges of the lattice
def is_inside(lattice, i, j):
    n = lattice.shape[0]
    if i < n and j < n and i >= 0 and j >= 0:
        return True
    else:
        return False

# Check for all four neighbours if they're inside the lattice and are a tree
def get_tree_neighbours(lattice, i, j):
    neighbours = []
    if is_inside(lattice, i-1, j) and lattice[i-1, j] == 1:
        neighbours.append((i-1, j))
    if is_inside(lattice, i, j-1) and lattice[i, j-1] == 1:
        neighbours.append((i, j-1))
    if is_inside(lattice, i, j+1) and lattice[i, j+1] == 1:
        neighbours.append((i, j+1))
    if is_inside(lattice, i+1, j) and lattice[i+1, j] == 1:
        neighbours.append((i+1, j))
    return neighbours

# One step of the simulation. Return False either when no more trees are burning or when there is a spanning cluster.
def step(lattice):
    n = lattice.shape[0]
    to_burn = set()
    for i in range(n):
        for j in range(n):
            if lattice[i, j] == 2:
                neighbours = get_tree_neighbours(lattice, i, j)
                to_burn.update(neighbours)
                lattice[i, j] = 3
    for site in to_burn:
        i, j = site
        lattice[i, j] = 2
    if len(to_burn) == 0 or is_spanning(lattice):
        return False
    else:
        return True

# Return true if the lattice contains a spanning cluster.
def is_spanning(lattice):
    if np.any(lattice[-1, :] == 3):
        return True
    else:
        return False
    

if __name__ == "__main__":
    images = []
    fig = plt.figure(figsize=(6, 6))
    args = create_arg_parser()

    # Create a square lattice of size n x n with tree density p
    grid = init_square_lattice(args.size, args.treeprobability, args.distribution)
    images.append(return_image(grid, 0))
    random_arson(grid)
    images.append(return_image(grid, 1))
    nstep = 1

    # TODO: the simulation is not working properly and stops early
    while step(grid):
        nstep += 1
        images.append(return_image(grid, nstep))
    ani = animation.ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=10000)

    # TODO: the gif is not saved properly
    ani.save(f'n={args.size}_p={int(100*args.treeprobability)}%_{args.distribution}.gif', dpi=80, writer='pillow') 

    plt.show()