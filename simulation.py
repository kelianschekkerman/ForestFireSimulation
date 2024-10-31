import numpy as np
import noise
import argparse
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation
from tqdm import tqdm
from scipy.optimize import curve_fit

### Change these to adjust iterations ###
max_iter = 20
max_subiter = 5
#########################################

# Take the arguments from the command line and return them
def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distribution", default='random', type=str,
                        help="Distribution of tree placement. Options are: random, semi-random, ordered. Default is random.")
    parser.add_argument("-t", "--treeprobability", default=0.6, type=float,
                        help="Probability of a tree being placed. Default is 0.6. Can range from 0.01 to 0.99. Only applicable for generating a single animation.")
    parser.add_argument("-s", "--size", default=100, type=int,
                        help="The size of the square lattice. Default is 100.")
    parser.add_argument("-ws", "--windspeed", default=0.1, type=float,
                        help="The speed of the wind. Default is 0.0, thus no wind. Can range from 0.0 to 0.3.")
    parser.add_argument("-wd", "--winddirection", default=None, type=str,
                        help="Direction of the wind. Options are: N, S, E, W, NE, NW, SE, SW. Default is no wind.")
    parser.add_argument("-w", "--weather", default='normal', type=str,
                        help="The weather conditions. Options are: dry, normal, wet. Default is normal.")
    parser.add_argument("-c", "--center", default='center', type=str,
                        help="Starting point of the fire in the lattice. Options are: center and random. Default is center.")
    parser.add_argument("-a", "--animation", default='animation', type=str,
                        help="Type of output. Options are: graph and animation. Default is graph.")
    args = parser.parse_args() 
    return args

# Return the values of the weather conditions
def get_weather_values(weather):
    if weather == 'dry':
        return 1.25
    elif weather == 'normal':
        return 1
    elif weather == 'wet':
        return 0.75

# Return the probabilities of neigbours catching fire based on the wind direction and wind speed
def get_wind_probabilities(args):   
    wind_speed = args.windspeed
    wind_direction = args.winddirection
    weather = get_weather_values(args.weather)

    if wind_direction == None:
        probabilties = [1, 1, 1, 1] # f.l.t.r. top, right, bottom, left
        return [x * weather for x in probabilties]
    elif wind_direction == 'N':
        probabilties = [0.25, 0.75, 1, 0.75]
        ws = [-wind_speed, wind_speed, wind_speed, wind_speed]
        weather_probabilities = [x * weather for x in probabilties]
        return [sum(x) for x in zip(ws, weather_probabilities)]
    elif wind_direction == 'S':
        probabilties = [1, 0.75, 0.25, 0.75]
        ws = [wind_speed, wind_speed, -wind_speed, wind_speed]
        weather_probabilities = [x * weather for x in probabilties]
        return [sum(x) for x in zip(ws, weather_probabilities)]
    elif wind_direction == 'E':
        probabilties = [0.75, 0.25, 0.75, 1]
        ws = [wind_speed, -wind_speed, wind_speed, wind_speed]
        weather_probabilities = [x * weather for x in probabilties]
        return [sum(x) for x in zip(ws, weather_probabilities)]
    elif wind_direction == 'W':
        probabilties = [0.75, 1, 0.75, 0.25]
        ws = [wind_speed, wind_speed, wind_speed, -wind_speed]
        weather_probabilities = [x * weather for x in probabilties]
        return [sum(x) for x in zip(ws, weather_probabilities)]
    elif wind_direction == 'NE':
        probabilties = [0.5, 0.5, 1, 1]
        ws = [-wind_speed, -wind_speed, wind_speed, wind_speed]
        weather_probabilities = [x * weather for x in probabilties]
        return [sum(x) for x in zip(ws, weather_probabilities)]
    elif wind_direction == 'NW':
        probabilties = [0.5, 1, 1, 0.5]
        ws = [-wind_speed, wind_speed, wind_speed, -wind_speed]
        weather_probabilities = [x * weather for x in probabilties]
        return [sum(x) for x in zip(ws, weather_probabilities)]
    elif wind_direction == 'SE':
        probabilties = [1, 0.5, 0.5, 1]
        ws = [wind_speed, -wind_speed, -wind_speed, wind_speed]
        weather_probabilities = [x * weather for x in probabilties]
        return [sum(x) for x in zip(ws, weather_probabilities)]
    elif wind_direction == 'SW':
        probabilties = [1, 1, 0.5, 0.5]
        ws = [wind_speed, wind_speed, -wind_speed, -wind_speed]
        weather_probabilities = [x * weather for x in probabilties]
        return [sum(x) for x in zip(ws, weather_probabilities)]

# Initialize the square lattice with trees based on the distribution type
def init_square_lattice(args, p):
    n = args.size
    type = args.distribution

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
    im = plt.imshow(lattice, cmap=cmap, norm=norm, animated=True, origin='lower')
    return [im]

def run_single_animation(wind_probabilities, args):
    images = []
    fig = plt.figure(figsize=(6, 6))

    # Create a square lattice of size n x n with tree density p
    grid = init_square_lattice(args, args.treeprobability)

    images.append(return_image(grid))
    arson(grid, args.center)
    images.append(return_image(grid))
    
    # Run the simulation and save the images for the animation
    while step(grid, wind_probabilities):
        images.append(return_image(grid))
    ani = animation.ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=10000)
    ani.save(f'gifs\{args.size}_{args.treeprobability}_{args.distribution}_{args.windspeed}_{args.winddirection}_{args.weather}_{args.center}.gif', dpi=80, writer='pillow') 
    plt.close(fig)
    # TODO: add variables in the gif next to just the file name

def find_closest_tree(lattice):
    n = lattice.shape[0]
    center = n//2
    min_distance = float('inf')
    closest_tree = None
    
    # Loop through the lattice to find the closest tree (value == 1)
    for i in range(n):
        for j in range(n):
            if lattice[i, j] == 1:
                # Calculate Euclidean distance from the center
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_tree = (i, j)
    
    return closest_tree

# Start the fire by setting a random tree on fire or the closest tree to the center
def arson(grid, center):
    n = grid.shape[0]

    if center == 'center':
        if grid[n//2, n//2] == 1:
            grid[n//2, n//2] = 2
            return  
        else:
            closest_tree = find_closest_tree(grid)
            grid[closest_tree[0], closest_tree[1]] = 2
            return
    elif center == 'random':
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
        if random.randint(0,100) <= (wind_probabilities[0]*100):
            neighbours.append((i+1, j))
    if is_inside(lattice, i, j+1) and lattice[i, j+1] == 1: # right neighbour
        if random.randint(0,100) <= (wind_probabilities[1]*100):
            neighbours.append((i, j+1))
    if is_inside(lattice, i-1, j) and lattice[i-1, j] == 1: # bottom neigbour
        if random.randint(0,100) <= (wind_probabilities[2]*100):
            neighbours.append((i-1, j)) 
    if is_inside(lattice, i, j-1) and lattice[i, j-1] == 1: # left neighbour
        if random.randint(0,100) <= (wind_probabilities[3]*100):
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
    if len(to_burn) == 0:
        return False
    else:
        return True
    
# Return if the perculation hit one of the sides of the square lattice
def is_percolating(lattice):
    if np.any(lattice[0, :] == 3) or np.any(lattice[:, 0] == 3) or np.any(lattice[-1, :] == 3) or np.any(lattice[:, -1] == 3):
        return True
    else:
        return False
    
# Define the sigmoid function to fit the data points
def sigmoid(x, max_val, x0, steepness, y_intercept, ):
    return max_val / (1 + np.exp(-steepness * (x - x0))) + y_intercept

# Create a graph with the data points and the fitted sigmoid curve
def add_to_graph(values, threshold, variable, color):

    # Define the x and y values of the data and save in numpy arrays
    x_values = np.array([i / len(values) for i in range(len(values))])
    y_values = np.array(values)

    # Fit the sigmoid curve to find the optimal values for our data
    opt_val, _ = curve_fit(sigmoid, x_values, y_values, maxfev=10000)

    # Extract the percolation threshold (x0 parameter in the sigmoid function)
    percolation_threshold = opt_val[1]

    # Calculate smooth x-values for the sigmoid curve
    x_smooth = np.linspace(min(x_values), max(x_values), 500)
    y_smooth = sigmoid(x_smooth, *opt_val)

    # Plot the original data and the fitted sigmoid curve
    # plt.scatter(x_values, y_values, s=30, c=color)
    plt.plot(x_smooth, y_smooth, c=color, label=f"{variable}")

    if threshold:
        # Add a vertical line for the percolation threshold
        plt.axvline(x=percolation_threshold, linestyle='--', c=color, label=f'Percolation threshold ≈ {percolation_threshold:.3f}')
    
# Save the graph
def save_graph(ylabel, title, sub_title, setting):
    plt.xlabel('Tree density')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'figures/{sub_title}_{setting}.png')
    plt.close()


# Main function to run the simulation
if __name__ == "__main__":
    args = create_arg_parser()

    # Overview of all possible variables for the simulation
    wind_directions = [None, 'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']
    wind_speeds = [0.1, 0.2, 0.3]
    weather_conditions = ['dry', 'normal', 'wet']
    location = ['center', 'random']
    
    settings = [weather_conditions, wind_directions, wind_speeds, location]

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']

    # Get the probabilities of the neighbours catching fire based on the wind direction, wind speed and weather
    wind_probabilities = get_wind_probabilities(args)
    np.clip(wind_probabilities, 0, 1)

    probability_step = 1.00000/max_iter

    # Just create a single animation if specified in the arguments
    if args.animation == 'animation':
        run_single_animation(wind_probabilities, args)
        exit()

    for setting in tqdm(settings):
        plt.figure(figsize=(12,6))

        # Reset the variables to the default values
        if setting == wind_directions:
            args.windspeed = 0.1
            args.weather = 'normal'
            args.center = 'center'
        elif setting == wind_speeds:
            args.winddirection = None
            args.weather = 'normal'
            args.center = 'center'
        elif setting == weather_conditions:
            args.winddirection = None
            args.windspeed = 0.1
            args.center = 'center'
        elif setting == location:
            args.winddirection = None
            args.windspeed = 0.1
            args.weather = 'normal'

        for variable in tqdm(setting):

            if setting == wind_directions:
                args.winddirection = variable
                wind_probabilities = get_wind_probabilities(args)
            elif setting == wind_speeds:
                args.windspeed = variable
                wind_probabilities = get_wind_probabilities(args)
            elif setting == weather_conditions:
                args.weather = variable
                wind_probabilities = get_wind_probabilities(args)
            elif setting == location:
                args.center = variable

            treeprobability = probability_step
            burnt_percentage = []
            percolating_percentage = []
            percolating_amount = 0
            burnt_trees_amount = 0

            for i in tqdm(range(max_iter - 1)): # probability does not start at 0, so we need max_iter - 1 steps
                for j in range(max_subiter):
                    percolating = False

                    # Create a square lattice of size n x n with tree density p
                    grid = init_square_lattice(args, treeprobability)
                    amount_of_trees = len(np.flatnonzero(grid == 1))
                    arson(grid, args.center)
                    
                    # Run the simulation and save the images for the animation
                    while step(grid, wind_probabilities):
                        pass

                    amount_of_trees_burnt = len(np.flatnonzero(grid == 3)) + len(np.flatnonzero(grid == 2))
                    percolating = is_percolating(grid)
                    percolating_amount += 1 if percolating else 0
                    burnt_trees_amount += (amount_of_trees_burnt / amount_of_trees)

                burnt_percentage.append(burnt_trees_amount / max_subiter)
                percolating_percentage.append(percolating_amount / max_subiter)
                percolating_amount = 0
                burnt_trees_amount = 0
                treeprobability += probability_step
        
            # Create the graphs for the percolating and burnt trees percentage
            add_to_graph(percolating_percentage, True, variable, colors[setting.index(variable)])
            # add_to_graph(burnt_percentage, False, variable)

        save_graph('Percolating percentage', 'Percolating percentage based on tree density', 'p', setting)
        # save_graph('Burnt trees percentage', 'Burnt trees percentage based on tree density', 'b', setting)