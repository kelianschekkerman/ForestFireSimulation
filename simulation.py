import numpy as np
from noise import pnoise2
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
    parser.add_argument("-a", "--animation", default='animation', type=str,
                        help="Type of output. Options are: graph and animation. Default is graph.")
    parser.add_argument("-s", "--size", default=100, type=int,
                        help="The size of the square lattice. Default is 100.")
    parser.add_argument("-d", "--distribution", default='random', type=str,
                        help="Distribution of tree placement. Options are: random, semi-random, ordered. Default is random.")
    parser.add_argument("-t", "--treeprobability", default=0.6, type=float,
                        help="Probability of a tree being placed. Default is 0.6. Can range from 0.01 to 0.99. Only applicable for generating a single animation.")
    parser.add_argument("-c", "--center", default='center', type=str,
                        help="Starting point of the fire in the lattice. Options are: center and random. Default is center.")
    parser.add_argument("-wd", "--winddirection", default=None, type=str,
                        help="Direction of the wind. Options are: N, S, E, W, NE, NW, SE, SW. Default is no wind.")
    parser.add_argument("-ws", "--windspeed", default=0.0, type=float,
                        help="The speed of the wind. Default is 0.0, thus no wind. Effective range is from 0.0 to 0.5.")
    parser.add_argument("-w", "--weather", default='normal', type=str,
                        help="The weather conditions. Options are: dry, normal, wet. Default is normal.")
    args = parser.parse_args() 
    return args

# Return the values of the weather conditions
def get_weather_values(weather):
    if weather == 'dry':   # trees are more likely to catch fire
        return 1.25
    elif weather == 'normal':
        return 1
    elif weather == 'wet': # trees are less likely to catch fire
        return 0.75

# Return the probabilities of neighbors catching fire based on the wind direction and wind speed
def get_wind_probabilities(args):   
    wind_speed = args.windspeed
    weather = get_weather_values(args.weather)
    
    # Define probabilities and wind adjustments for each direction
    direction_map = {
        None:   ([1, 1, 1, 1],                [0, 0, 0, 0]),  # No wind
        'N':    ([0.25, 0.75, 1, 0.75],       [-wind_speed, wind_speed, wind_speed, wind_speed]), # f.l.t.r (top, right, bottom, left)
        'S':    ([1, 0.75, 0.25, 0.75],       [wind_speed, wind_speed, -wind_speed, wind_speed]),
        'E':    ([0.75, 0.25, 0.75, 1],       [wind_speed, -wind_speed, wind_speed, wind_speed]),
        'W':    ([0.75, 1, 0.75, 0.25],       [wind_speed, wind_speed, wind_speed, -wind_speed]),
        'NE':   ([0.5, 0.5, 1, 1],            [-wind_speed, -wind_speed, wind_speed, wind_speed]),
        'NW':   ([0.5, 1, 1, 0.5],            [-wind_speed, wind_speed, wind_speed, -wind_speed]),
        'SE':   ([1, 0.5, 0.5, 1],            [wind_speed, -wind_speed, -wind_speed, wind_speed]),
        'SW':   ([1, 1, 0.5, 0.5],            [wind_speed, wind_speed, -wind_speed, -wind_speed])
    }
    
    # Retrieve probabilities and wind adjustments for the given direction
    if wind_speed == 0.0:
        probabilities, ws = direction_map.get(None)
    else:
        probabilities, ws = direction_map.get(args.winddirection)
    # Adjust probabilities by weather and add wind effects
    weather_probabilities = [p * weather for p in probabilities]
    return [sum(x) for x in zip(ws, weather_probabilities)]

# Map a value from one range to another
def map_value(x, old_min=0, old_max=1, new_min=2, new_max=50):
    mapped_value = new_min - ((x - old_min) / (old_max - old_min)) * (new_max - new_min)
    return int(mapped_value)

# Initialize the square lattice with trees based on the distribution type
def init_square_lattice(args, p):
    n = args.size
    type = args.distribution

    # random tree distribution on the lattice
    if type == 'random': 
        lattice = np.random.choice([0, 1], size=(n, n), p=[1-p, p])

    # ordered lines on the lattice
    elif type == 'ordered': 
        d = map_value(p) # distance between lines based on tree density
        if d < 2:
            d = 2 # slightly hardcoded to prevent division by zero
        
        lattice = np.zeros((n, n))
        for i in range(n):
            if i % d == 0:
                lattice[i, :] = 1
            for j in range(n):
                if j % d == 0:
                    lattice[:, j] = 1

    # random tree distribution based on perlin noise
    elif type == 'semi-random': 
        lattice = np.zeros((n, n))
        scale = 1.9 
        for i in range(n):
            for j in range(n):
                 noise_value = pnoise2(i / scale, j / scale)
                 normalized_value = (noise_value + 1) / 2
                 lattice[i, j] = 1 if normalized_value < p else 0

    return lattice

# Return a single image used in a matplotlib animation
def return_image(lattice, title="Forest fire"):
    cmap = ListedColormap(['w', 'g', 'r', 'k'])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    plt.title(title)
    plt.axis('off')
    im = plt.imshow(lattice, cmap=cmap, norm=norm, animated=True, origin='lower')
    return [im]

# Run a single animation based on the given arguments
def run_single_simulation(wind_probabilities, args):
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

    if args.distribution == 'random':
        ani.save(f'gifs\{args.size}_{args.distribution}_{args.treeprobability}_{args.winddirection}_{args.windspeed}_{args.weather}_{args.center}.gif', dpi=80, writer='pillow') 
    else:
        ani.save(f'gifs\{args.size}_{args.distribution}_{args.winddirection}_{args.windspeed}_{args.weather}_{args.center}.gif', dpi=80, writer='pillow')
    plt.close(fig)

# Run a single animation based on the given arguments
def run_simulation(args, wind_probabilities, treeprobability):
    percolating_amount = 0
    burnt_trees_amount = 0

    for _ in range(max_subiter):
        grid = init_square_lattice(args, treeprobability)
        amount_of_trees = len(np.flatnonzero(grid == 1))
        arson(grid, args.center)

        # Run the simulation steps
        while step(grid, wind_probabilities):
            pass

        amount_of_trees_burnt = len(np.flatnonzero(grid == 3)) + len(np.flatnonzero(grid == 2))
        if is_percolating(grid):
            percolating_amount += 1
        burnt_trees_amount += (amount_of_trees_burnt / amount_of_trees)
    
    return percolating_amount, burnt_trees_amount

# Loop through the lattice to find the closest tree (value == 1)
def find_closest_tree(lattice):
    n = lattice.shape[0]
    center = n//2
    min_distance = float('inf')
    closest_tree = None
    
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
        else:
            closest_tree = find_closest_tree(grid)
            grid[closest_tree[0], closest_tree[1]] = 2
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
def add_to_graph(values, scatter, threshold, variable, color):

    # Define the x and y values of the data and save in numpy arrays
    x_values = np.array([i / len(values) for i in range(len(values))])
    y_values = np.array(values)

    if scatter:
        plt.scatter(x_values, y_values, s=20, c=color, label=f"{variable}")

    else:
        # Fit the sigmoid curve to find the optimal values for our data
        opt_val, _ = curve_fit(sigmoid, x_values, y_values, maxfev=10000)

        # Extract the percolation threshold (x0 parameter in the sigmoid function)
        percolation_threshold = opt_val[1]

        # Calculate smooth x-values for the sigmoid curve
        x_smooth = np.linspace(min(x_values), max(x_values), 500)
        y_smooth = sigmoid(x_smooth, *opt_val)

        # Plot the original data and the fitted sigmoid curve
        plt.plot(x_smooth, y_smooth, c=color, label=f"{variable}")

        if threshold:
            # Add a vertical line for the percolation threshold
            plt.axvline(x=percolation_threshold, linestyle='--', c=color, label=f'Percolation threshold ≈ {percolation_threshold:.3f}')
    
# Save the graph
def save_graph(ylabel, title, sub_title, setting, distribution):
    plt.xlabel('Tree density')
    plt.ylabel(ylabel)
    plt.title(title + f' and {setting}')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'figures/{sub_title}_{distribution}_{setting}.png')
    plt.close()

# Reset the default values for the given setting
def reset_defaults(setting):
    for key, value in defaults[setting].items():
        setattr(args, key, value)

# Update the arguments based on the variable and setting
def update_args_for_variable(setting, variable):
    if setting == "wind_directions":
        args.winddirection = variable
    elif setting == "wind_speeds":
        args.windspeed = variable
    elif setting == "weather_conditions":
        args.weather = variable
    elif setting == "location":
        args.center = variable
    return get_wind_probabilities(args)


# Main function to run the simulation
if __name__ == "__main__":
    args = create_arg_parser()

    # Overview of all possible variables for the simulation
    wind_directions = [None, 'N', 'NE']
    wind_speeds = [0.0, 0.1, 0.2, 0.3]
    weather_conditions = ['dry', 'normal', 'wet']
    location = ['center', 'random']
    distributions = ['random', 'ordered']
    probability_step = 1.00000/max_iter

    settings = [wind_speeds, wind_directions, weather_conditions, location]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']

    wind_probabilities = get_wind_probabilities(args)
    np.clip(wind_probabilities, 0, 1)

    # Just create a single animation if specified in the arguments
    if args.animation == 'animation':
        run_single_simulation(wind_probabilities, args)
        exit()

    # Dictionary of default values for different settings
    defaults = {
        "wind_directions":    {'windspeed': 0.1, 'weather': 'normal', 'center': 'center'},
        "wind_speeds":        {'winddirection': 'N', 'weather': 'normal', 'center': 'center'},
        "weather_conditions": {'winddirection': None, 'windspeed': 0.0, 'center': 'center'},
        "location":           {'winddirection': None, 'windspeed': 0.0, 'weather': 'normal'}
    }

    # Loop through all distributions and settings to generate graphs
    for distribution in tqdm(distributions, desc='Distributions'):
        args.distribution = distribution

        # Loop through all settings and variables to generate graphs
        for setting in tqdm(settings, desc='Settings'):
            burnt_percentages = []
            percolating_percentages = []
            setting_name =  "wind_directions" if setting == wind_directions else \
                            "wind_speeds" if setting == wind_speeds else \
                            "weather_conditions" if setting == weather_conditions else \
                            "location"
            reset_defaults(setting_name)

            # Loop through all variables for the given setting
            for variable in tqdm(setting, desc=setting_name):
                wind_probabilities = update_args_for_variable(setting_name, variable)
                treeprobability = probability_step
                burnt_percentage = []
                percolating_percentage = []
                
                # Run main simulation loop and keep track of the burnt and percolating percentages
                for i in tqdm(range(max_iter - 1), desc='Run the simulation'):
                    percolating_amount, burnt_trees_amount = run_simulation(args, wind_probabilities, treeprobability)
                    burnt_percentage.append(burnt_trees_amount / max_subiter)
                    percolating_percentage.append(percolating_amount / max_subiter)
                    treeprobability += probability_step

                percolating_percentages.append(percolating_percentage)
                burnt_percentages.append(burnt_percentage)

            # Generate both the percolating and burnt trees graphs
            if distribution == 'random':
                plt.figure(figsize=(12, 6))
                for idx, percolating_percentage in enumerate(percolating_percentages):
                    add_to_graph(percolating_percentage, False, True, setting[idx], colors[idx])
                save_graph('Percolating percentage', 'Percolating percentage based on tree density', 'p', setting_name, distribution)

                plt.figure(figsize=(12, 6))
                for idx, burnt_percentage in enumerate(burnt_percentages):
                    add_to_graph(burnt_percentage, False, False, setting[idx], colors[idx])
                save_graph('Burnt trees percentage', 'Burnt trees percentage based on tree density', 'b', setting_name, distribution)

            # Generate only the burnt trees graph
            else:
                plt.figure(figsize=(12, 6))
                for idx, burnt_percentage in enumerate(burnt_percentages):
                    add_to_graph(burnt_percentage, True, False, setting[idx], colors[idx])
                save_graph('Burnt trees percentage', 'Burnt trees percentage based on tree density', 'b', setting_name, distribution)