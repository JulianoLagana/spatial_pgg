# Spatial Public Goods Game

Repository for developing a spatial simulation of the public goods game. Work in progress.

### Instructions for using this repo
These instructions have been tested for Ubuntu 18.04.03.
- Create a new virtual environment with Python 3.6.9 (`conda create -n spatial_pgg_env python=3.6.9` if you're using conda).
- Activate the virtual environment you created (`conda activate spatial_pgg_env`).
- Install the required packages with `pip install -r requirements.txt`.


### Available simulations
- `well_mixed_simulation.py` runs a simulation of the public goods game in a well-mixed population (no spatial aspect here).
- `small_world_simulation.py` runs a simulation of the public goods game where players are connected according to a small-world graph.
