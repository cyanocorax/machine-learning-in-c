# Overview
A multi-layered perceptron (aka. MLP) implemented in C.

# Usage
Just use make command and run it.
```
cd mlp
make
./app
```

# Parameters
You can changes the parameters below according to your own demands.
## Macros
- EPOCHS: The number of iterations (learnings).
- LR: The learning rate, that can slow down or speed up the perceptron's learning.
## Local variables
- sizes: The sizes for each layer.
- inputs: The inputs to the perceptron.
- targets: The teacher data.
## Activation function
- You can pass an activation function to the function mlp_set_activation(), by choosing one of them as follows:
 - ACT_RELU: relu function.
 - ACT_SIGMOID: sigomoid function.
 - ACT_TANH: tanh function.
