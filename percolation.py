import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation
from scipy.stats import bernoulli
from enum import IntEnum
import time
from typing import List, Tuple, Dict
from numba import jit, vectorize
import pandas as pd

class SiteStatus(IntEnum):
    EMPTY = 0
    TREE = 1
    BURNING = 2
    BURNT = 3

BURNT = int(SiteStatus.BURNT) # hack that is necessary for numba's jit to work - it doesn't like enums...


def plot_lattice(lattice: np.ndarray, nstep: int):
    """Plot a specific lattice instance. nstep is used in the title of the image"""
    plt.figure(figsize=(6,6))
    cmap = ListedColormap(['w', 'g', 'r', 'k'])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    plt.matshow(lattice, cmap=cmap, norm=norm)
    plt.savefig(f'nstep={nstep}')


def return_image(lattice, nstep) -> List:
    """Return a single image used in a matplotlib animation"""
    cmap = ListedColormap(['w', 'g', 'r', 'k'])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    im = plt.imshow(lattice, cmap=cmap, norm=norm, animated=True)
    return [im]


def init_square_lattice(length: int, occupation_prob: float) -> np.ndarray:
    lattice = bernoulli.rvs(occupation_prob, size=length*length)
    return lattice.reshape((length, length))


def start_fire(lattice: np.ndarray):
    has_trees = lattice[0,:] == SiteStatus.TREE
    lattice[0, has_trees] = SiteStatus.BURNING
    return lattice

@jit(nopython=True)
def is_inside(lattice: np.ndarray, i: int, j: int) -> bool:
    """Return true if site at i,j is inside of the lattice"""
    n = lattice.shape[0]
    if i < n and j < n and i >= 0 and j >= 0:
        return True
    else:
        return False

@jit(nopython=True)
def get_tree_neighbours(lattice: np.ndarray, i: int, j:int) -> List[Tuple[int,int]]:
    """Check for all four neighbours if they're inside the lattice and are a tree"""
    neighbours = []
    if is_inside(lattice, i-1, j) and lattice[i-1, j] == SiteStatus.TREE:
        neighbours.append((i-1, j))
    if is_inside(lattice, i, j-1) and lattice[i, j-1] == SiteStatus.TREE:
        neighbours.append((i, j-1))
    if is_inside(lattice, i, j+1) and lattice[i, j+1] == SiteStatus.TREE:
        neighbours.append((i, j+1))
    if is_inside(lattice, i+1, j) and lattice[i+1, j] == SiteStatus.TREE:
        neighbours.append((i+1, j))
    return neighbours

@jit(nopython=True)
def step(lattice) -> bool:
    """One step of the simulation. Return False either when no more trees are burning
    or when there is a spanning cluster."""
    n = lattice.shape[0]
    to_burn = set()
    for i in range(n):
        for j in range(n):
            if lattice[i, j] == SiteStatus.BURNING:
                neighbours = get_tree_neighbours(lattice, i, j)
                to_burn.update(neighbours)
                lattice[i, j] = SiteStatus.BURNT
    for site in to_burn: # TODO optimize: Loop over lattice is faster than random memory accesses
        i, j = site
        lattice[i, j] = SiteStatus.BURNING
    if len(to_burn) == 0 or is_spanning(lattice):
        return False
    else:
        return True

def plot_animation_single_instance(n, p):
    """Plots the animation for a given n, p lattice."""
    images = []
    fig = plt.figure()
    l = init_square_lattice(n, p)
    images.append(return_image(l, 0))
    start_fire(l)
    images.append(return_image(l, 1))
    nstep = 1
    while step(l):
        nstep += 1
        images.append(return_image(l, nstep))
    ani = animation.ArtistAnimation(fig, images, interval=100, blit=True,
                                repeat_delay=10000)
    ani.save(f'n={n}_p={int(100*p)}%.gif', dpi=80, writer='imagemagick')
    plt.show()

@jit(nopython=True)
def is_spanning(lattice: np.ndarray) -> bool:
    """Return true if the lattice contains a spanning cluster."""
    if np.any(lattice[-1, :] == BURNT):
        return True
    else:
        return False

def run_simulation(n: int, p: float) -> Dict:
    """Run one instance of the simulation for a given n and p.
    Return a dictionary with results."""
    l = init_square_lattice(n, p)
    start_fire(l)
    n_step = 1
    while step(l):
        n_step += 1
    return {'n': n, 'p': p, 'is_spanning': is_spanning(l), 'shortest_path': n_step}


if __name__ == "__main__":
    np.random.seed(1)
    n = 50
    p = 0.59
    plot_animation_single_instance(n, p)

    # uncomment below to produce statistics
    '''
    n_repetitions = 1000
    res = []
    n = 400
    start = time.time()
    for p in np.arange(0.55, 0.63, 0.01):
        print(p)
        for i in range(n_repetitions):
            res.append(run_simulation(n, p))
    end = time.time()
    print(f'elapsed = {end-start}')
    df = pd.DataFrame(res)
    df.to_csv('results_critical.csv')
    '''