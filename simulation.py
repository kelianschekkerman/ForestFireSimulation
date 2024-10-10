import numpy as np
import noise
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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

def init_square_lattice(n, p, type):
    if type == 'random': # random tree distribution on the lattice
        lattice = np.random.choice([0, 1], size=(n, n), p=[1-p, p])
    elif type == 'ordered': # vertical tree lines on the lattice
        lattice = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j % 2 == 0:
                    lattice[i, j] = 1
    elif type == 'semi-random': # random tree distribution based on perlin noise
        lattice = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                result = noise.pnoise2(i/n, j/n, octaves=10, persistence=1, lacunarity=20.0)
                lattice[i, j] = 0 if result < 0 else 1
    return lattice

def show_grid(grid, title="Forest fire"):
    plt.figure(figsize=(6, 6))
    cmap = ListedColormap(['w', 'g', 'r', 'k'])
    plt.imshow(grid, cmap=cmap, interpolation='none')
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    args = create_arg_parser()
    # Create a square lattice of size n x n with tree density p
    lattice = init_square_lattice(args.size, args.treeprobability, args.distribution)
    show_grid(lattice)