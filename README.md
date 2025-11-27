# Is the agent Bayesian? How?
This repo contains the experiments we conducted for our Bayesian Modeling of Brain and Behavior Course project. Our task consisted in -----, which could be assessed at 5 different levels of difficulty.

## How to use?
Clone the repository
uv sync to create virtual environment from pyproject.toml

## Src
- data_gen contains the code to generate --- 
- notebooks contains explanation notebooks of the experiments for the corresponding level (1 to 5). One notebook per level
Run tests from the notebook, make sure to change the mean/variance psychometric data location, 

## Data Folders
- test/lvl*/lvl*_mean: output of data_gen_meanPrior.py 
- test/lvl*/lvl*_var: output of data_gen_varPrior.py
- ex_data/lvl*/lvl*_mean: output of the experiments for the mean estimation
- ex_data/lvl*/lvl*_var: output of the experiments for the var estimation
- combined_psychometric/lvl*/*: output of concat_*.py which outs the required .csv for psychometric 

## Authors:
Sarush & Jay & Rowan & Cami


