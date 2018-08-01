# Solution to the first project


### Solution: Evolution Strategies with Neural networks
The idea is based on two methods: deep neural nets and [evolution strategies](https://blog.openai.com/evolution-strategies/).  
For the neural net, a LSTM network is used to output a sequence of elements. Given a needed list of n elements, such as: ['S2_SrpR', 'Q2_QacR', 'H1_HlyIIR', 'R1_PsrA', 'P3_PhlF'], the network will first output 'S2_SrpR', then take as input 'S2_SrpR' and output 'Q2_QacR', then take as input ['S2_SrpR', 'Q2_QacR'] and output 'R1_PsrA'.  
  

To train the network, evolution strategies are used. In a few words, noise is added to the parameters of the neural network. Then, the net produces a given number of sequences to solve the given problem, which are scored according to the library provided for the BDAthlon. This score is then given back to the network to modifiy its parameters such that the score is increased.  
  
The Python library [Keras](https://keras.io/) is used to create the neural net and [evostra](https://github.com/alirezamika/evostra) for the evolution strategies.  

Note: [reinforcement learning with deep neural nets](https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning/) could also have be used here. However, to run this kind of training algorithm there is a huge need of computational power, namely at least one GPU. Thus, evolution strategies seemed to be more suited to get to a broader audience. Moreover, evoution strategies have proven to be effective (cf link above).  

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
To run the script to find the solution, you should:  
1. clone the library provided for the competition at the same level as this repo  
2. then cd into 2018-Gotteron-1/solution/  
3. run the following script with the arguments described below.  
  
Arguments:  
--path_json: path to the file you want to find the solution to.  
--path_library: path to the library file (i.e. genetic_gate_library.json)  
--name: name of your ran. Will be used to name the output file were the solution will be saved.  
--n_epoch: the number of time the algorithm iterates.  
note: the best score will be print in the terminal as well as put in the filename. 
  
Here is an example:

```
python solve_problem_1.py --path_json '../../genetic_logic_synthesis/genetic_circuit_scoring/example/majority_mapping.json' --path_library '../../genetic_logic_synthesis/genetic_circuit_scoring/example/genetic_gate_library.json' --name majority --n_epoch 5
```

### Results
after 5 epochs for each circuit (training time of 5 minutes; more details below):  
majority_netlist.json: ~0.91  
multiplexer_netlist.json: ~0.99  
rule_30_netlist.json: ~0.91

### Note on the performance of the algorithm
with 5 epochs, the algorithm should take approximately 5 minutes on a machine equivalent to a MacBook Pro (Retina, 13-inch, Early 2015) (Processor 2.7 GHz Intel Core i5, Memory 8 GB 1867 MHz DDR3).  
However, if you want/have the time to run the algorithm longer, you can increase the number of epochs.  

Also, note that the time is displayed at the end of the script run. Moreover, the results are sotchastic (dictionnary are saved in solution/). So, the solutions files with their corresponding score are just an example of results found after one run of 5 epochs for each circuits.

### Future work
It would be very interesting to explore the use of [transfer learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/). Indeed, we can imagine that the features learned to solve a given circuit will be usefull to solve another circuit. Thus, this would potentially increased the results as well as diminishing the training time.
