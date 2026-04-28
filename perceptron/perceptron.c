#include "perceptron.h"

void perceptron_forward(double *inputs, double *weights, int n_inputs, double bias, double (*activate)(double sum), double *output) {
  int i;
  double sum;
  for(i = 0, sum = 0; i < n_inputs; i++)
    sum += inputs[i] * weights[i];
  *output = activate(sum + bias);
}

void perceptron_backward(double *inputs, double *weights, int n_inputs, double *bias, double error, double learning_rate) {
  int i;
  for(i = 0; i < n_inputs; i++)
    weights[i] = weights[i] + learning_rate * error * inputs[i];
  *bias = *bias + learning_rate * error;
}
