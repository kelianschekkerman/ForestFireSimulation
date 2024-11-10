# Forest Fire Simulation

This project simulates the spread of a forest fire using a simple model. The simulation is implemented in `simulation.py`.

## Features

- Simulates forest fire spread over a grid.
- Configurable parameters for tree density, forest type, and grid size.
- Create graphs for percolation percentage and burnt tree percentage

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

There are several input parameters. These parameters are optional, you can find the parameters by running:
```sh
python simulation.py -h
```

## Acknowledgements

- Inspired by the forest fire model code from https://gitlab.com/stunderline/forestfire/tree/master. 
- Also inspired by the slides on percolation in the Modelling and Simulation course.
