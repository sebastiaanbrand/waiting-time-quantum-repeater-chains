# Waiting time and fidelity in quantum repeater chains

This repository contains implementations of two algorithms (a Monte Carlo algorithm and a deterministic algorithm) for computing the probability distribution of waiting time for the delivery of entanglement in quantum repeater chains. These algorithms can also compute the Werner parameters / fidelities of the produced entanglement.
[TODO: reference paper]


## Getting started

### Prerequisites
This software was written for `Python 3`. The following packages are required:
```
numpy, scipy, matplotlib
```

### Installation
Just download/clone the repository.

### Example usage
The script `main.py` contains an example of how to use the algorithms and can be ran with the command `python main.py`.  This script computes and plots the probability distribution of waiting time for the SWAP-ONLY repeater chain protocol, using both the numerical calculation algorithm, as well as the Monte Carlo algorithm, for some parameters which are set and can be changed in the script. An example of how the Werner parameters can be computed along side the waiting time is also included.

## Files overview
- `main.py` contains an example script how to call the relevant functions in the next file:
- `repeater_chain_analyzer.py` contains implementations of both the deterministic computation algorithm, and the Monte Carlo algorithm.
- `probability_tools.py` contains a number of probability theory releated functions used by the deterministic algorithm.
- `werner_tools.py` contains functions related to computing the Werner parameter of states given certain operations, and is used by both algorithms.
- `plotting_tools.py` contains some functions to plot the probability distributions and the Werner parameters or fidelities.




