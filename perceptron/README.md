# Overview
A perceptron (single-layered perceptron) implemented in C.

# Usage
Just use make command and run it.
```
cd perceptron
make
./app
```

# Parameters
You can changes the parameters below according to your own demands.
## Macros
- N_INPUTS: The number of inputs, that must be the same # of inputs below.
- N_ITERS: The number of iterations of learning, aka, epochs.
- LR: The learning rate, that can slow down or speed up the perceptron's learning.
## Local variables
- inputs: The inputs to the perceptron.
- t_output: The teacher data that is be compared with the output of the perceptron.
## Activation function
- sigmoid: The program uses sigmoid function as an activation function. You can change it with another one, the signature of which is 'double some_func(double)'.
