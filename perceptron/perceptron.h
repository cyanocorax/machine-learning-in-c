#ifndef _PERCEPTRON_
#define _PERCEPTRON_

void perceptron_forward(double *inputs, double *weights, int n_inputs, double bias, double (*activate)(double sum), double *output);
void perceptron_backward(double *inputs, double *weights, int n_inputs, double *bias, double error, double learning_rate);

#endif
