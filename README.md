# Is the agent Bayesian? How?
This repo contains the experiments we conducted for our Bayesian Modeling of Brain and Behavior Course project. Our task consisted in estimate different hidden cognitive variables of simulated bayesian agents. We had to estimate a different set of variables for each of the 5 proposed levels of difficulty. 

## How to use?
Clone the repository
uv sync to create virtual environment from pyproject.toml

run files in level 1 in order, respecting the stops to upload and download data from the website. Then refer to the notebook for step by step explanation.

similarly, for level 2, level 3, level 4,...

*I think I'm mixing up the names experiment and level in some parts. i'll come back to that later.

## Src
```bash
src/
├── bmpe/
│   ├── utils.py
│   ├── experiment_1/
│   │   ├── 01_L1_data_gen_meanPrior.py        # generates simulated data for the mean-prior estimates
│   │   ├── 02_L1_concat_mean.py               # concatenates output from website
│   │   ├── 03_L1_get_mean_estimate.py         # computes the mean estimate from psychometric fit using probit
│   │   ├── 04_L1_data_gen_varPrior.py         # generates data for variance-prior estimation
│   │   └── 05_L1_concat_var.py                # concatenates variance estimation outputs from website
│   ├── level_2/
│   │   └── 01_L2_data_gen_meanPrior.py
│   └── __pycache__/                           # python bytecode (ignored)
└── BMPE.egg-info/                              # auto-generated when package is installed
```

## Data Folders
```bash
data/
├── experiment_1/
│   ├── website_input/               # Stimuli / settings fed to the online experiment
│   │   ├── mean/
│   │   └── variance/
│   └── website_output/              # Downloaded results from participants
│       ├── raw/                     # Raw CSVs, exactly as downloaded; never modified
│       └── processed/               # Cleaned or aggregated data ready for analysis
│
└── level_2/                         # Data for level-2 experiments
```


## Authors:
Sarush & Jay & Rowan & Cami (in no particular order)


