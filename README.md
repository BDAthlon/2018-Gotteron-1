# Solution to the first project


### Solution: Evolution Strategies with Neural networks


### Prerequisites

The cleaner way to install the prerequisites is to create a conda virtual environment. 

Firt, you can install conda by following the instructions here: https://docs.anaconda.com/anaconda/install/  

Or update it if you already have it installed:

```
conda update conda
```
And then create a virtual env from the yml file:

```
conda env create -f environment.yml

```

You can now activate your virtual env:

```
source activate bda
```

### Run the algorithm


```
python solve_problem_1.py --path_json '../../genetic_logic_synthesis/genetic_circuit_scoring/example/majority_mapping.json' --path_library '../../genetic_logic_synthesis/genetic_circuit_scoring/example/genetic_gate_library.json' --name majority --n_epoch 5
```
