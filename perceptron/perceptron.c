#include "perceptron.h"

perceptron *perceptron_alloc(const int n_inputs) {
  perceptron *p;

  p = (perceptron *)malloc(sizeof(perceptron));
  p->weights = (double *)malloc(sizeof(double) * n_inputs);
  p->n_inputs = n_inputs;
  return p;
}

void perceptron_free(perceptron *p) {
  free(p->weights);
  free(p);
}

void perceptron_init(perceptron *p) {
  int i;

  for(i = 0; i < p->n_inputs; i++)
    p->weights[i] = (double)(rand() % 101) / 100.0;
  p->bias = (double)(rand() % 101) / 100.0;
}

void perceptron_forward(perceptron *p, const double *inputs, double (*activate)(double sum), double *output) {
  int i;
  double sum;

  for(i = 0, sum = 0; i < p->n_inputs; i++)
    sum += inputs[i] * p->weights[i];
  *output = activate(sum + p->bias);
}

void perceptron_backward(perceptron *p, const double *inputs, const double error, const double learning_rate) {
  int i;

  for(i = 0; i < p->n_inputs; i++)
    p->weights[i] = p->weights[i] + learning_rate * error * inputs[i];
  p->bias = p->bias + learning_rate * error;
}
