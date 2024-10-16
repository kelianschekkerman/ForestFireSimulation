# Forest Fire Simulation

This project simulates the spread of a forest fire using a simple model. The simulation is implemented in `simulation.py`.

## Features

- Simulates forest fire spread over a grid.
- Configurable parameters for tree density, forest type, and grid size.

## Requirements

Requirements can be found in requirements.txt. You can install them by running:

```sh
pip install -r requirements.txt
```

## Usage

Run the simulation with default parameters:
```sh
python simulation.py
```

There are several imput parameters. These parameters are optional, you can find the parameters by running:
```sh
python simulation.py -h
```

## Project status
In this version of the code, we have implemented a grid representing a forest. Trees can be placed in this forest using different distributions. We have also implemented a starting point for the fire, this point is chosen randomly from all cells representing a tree. Our next step will be adding the spreading of fire to neighbouring trees. After that, we will focus on simulating the wind direction.

## Acknowledgements

- Inspired by the forest fire model code from https://gitlab.com/stunderline/forestfire/tree/master. 
- Also inspired by the slides on percolation in the Modelling and Simulation course.
