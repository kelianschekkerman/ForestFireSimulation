import numpy as np
import noise
import argparse
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation

# Take the arguments from the command line and return them
# TODO: add a catch for invalid arguments
def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distribution", default='random', type=str,
                        help="Distribution of tree placement. Options are: random, semi-random, ordered")
    parser.add_argument("-t", "--treeprobability", default=0.6, type=float,
                        help="Probability of a tree being placed.")
    parser.add_argument("-s", "--size", default=100, type=int,
                        help="The size of the square lattice.")
    parser.add_argument("-ws", "--windspeed", default=100, type=int,
                    help="The speed of the wind.")
    parser.add_argument("-wd", "--winddirection", default=None, type=str,
                        help="Direction of the wind. Options are: N, S, E, W, NE, NW, SE, SW. If not specified, there is no wind.")
    parser.add_argument("-w", "--weather", default='normal', type=str,
                help="The weather conditions. Options are: dry, normal, wet.")
    args = parser.parse_args() 
    return args

# Return the probabilities of neigbours catching fire based on the wind direction and wind speed
# TODO: add a formula for the probabilities based on the wind speed and direction
def get_wind_probabilities(wind_speed, wind_direction, weather):   
    if wind_direction == None:
        probabilties = [1, 1, 1, 1] # f.l.t.r. top, right, bottom, left
        return [x * weather for x in probabilties]
    elif wind_direction == 'N':
        probabilties = [0.25, 0.7, 1, 0.7]
        return [x * weather for x in probabilties]
    elif wind_direction == 'S':
        probabilties = [1, 0.7, 0.25, 0.7]
        return [x * weather for x in probabilties]
    elif wind_direction == 'E':
        probabilties = [0.7, 0.25, 0.7, 1]
        return [x * weather for x in probabilties]
    elif wind_direction == 'W':
        probabilties = [0.7, 1, 0.7, 0.25]
        return [x * weather for x in probabilties]
    elif wind_direction == 'NE':
        probabilties = [0.1, 0.1, 0.3, 0.5]
        return [x * weather for x in probabilties]
    elif wind_direction == 'NW':
        probabilties = [0.1, 0.5, 0.3, 0.1]
        return [x * weather for x in probabilties]
    elif wind_direction == 'SE':
        probabilties = [0.5, 0.1, 0.1, 0.3]
        return [x * weather for x in probabilties]
    elif wind_direction == 'SW':
        probabilties = [0.3, 0.1, 0.5, 0.1]
        return [x * weather for x in probabilties]

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
    plt.title(title)
    plt.axis('off')
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
def get_tree_neighbours(lattice, i, j, wind_probabilities):
    neighbours = []
    if is_inside(lattice, i+1, j) and lattice[i+1, j] == 1: # top neighbour
        if random.randint(0,100) < (wind_probabilities[0]*100):
            neighbours.append((i+1, j))
    if is_inside(lattice, i, j+1) and lattice[i, j+1] == 1: # right neighbour
        if random.randint(0,100) < (wind_probabilities[1]*100):
            neighbours.append((i, j+1))
    if is_inside(lattice, i-1, j) and lattice[i-1, j] == 1: # bottom neigbour
        if random.randint(0,100) < (wind_probabilities[2]*100):
            neighbours.append((i-1, j)) 
    if is_inside(lattice, i, j-1) and lattice[i, j-1] == 1: # left neighbour
        if random.randint(0,100) < (wind_probabilities[3]*100):
            neighbours.append((i, j-1))
    return neighbours

# One step of the simulation. Return False either when no more trees are burning or when there is a spanning cluster.
def step(lattice, wind_probabilities):
    n = lattice.shape[0]
    to_burn = set()
    for i in range(n):
        for j in range(n):
            if lattice[i, j] == 2:
                neighbours = get_tree_neighbours(lattice, i, j, wind_probabilities)
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
    if np.any(lattice[-1, :] == 4):
        return True
    else:
        return False
    

if __name__ == "__main__":
    args = create_arg_parser()

    if args.weather == 'dry':
        weather = 1.5
    elif args.weather == 'normal':
        weather = 1
    elif args.weather == 'wet':
        weather = 0.5

    burnt_percentage = []

    # TODO: Add a loop for multiple iterations to normalise the results
    images = []
    fig = plt.figure(figsize=(6, 6))

    # Create a square lattice of size n x n with tree density p
    grid = init_square_lattice(args.size, args.treeprobability, args.distribution)

    images.append(return_image(grid))
    random_arson(grid)
    images.append(return_image(grid))

    # Get the probabilities of the neighbours catching fire based on the wind direction, wind speed and weather
    wind_probabilities = get_wind_probabilities(args.windspeed, args.winddirection, weather)

    # Run the simulation and save the images for the animation
    while step(grid, wind_probabilities):
        images.append(return_image(grid))
    ani = animation.ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=10000)
    ani.save(f'gifs\{args.size}_{args.treeprobability}_{args.distribution}_{args.windspeed}_{args.winddirection}_{args.weather}.gif', dpi=80, writer='pillow') 

    # plt.show() # Uncomment to show the animation
    # TODO: calculate the percentage of trees burned down from the total number of trees
    amount_of_trees = len(np.argwhere(grid == 1))
    amount_of_trees_burnt = len(np.argwhere(grid == 3))
    print(amount_of_trees, amount_of_trees_burnt)