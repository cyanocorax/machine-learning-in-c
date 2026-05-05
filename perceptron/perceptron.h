#ifndef _PERCEPTRON_
#define _PERCEPTRON_

#include <stdlib.h>

typedef struct {
  double *weights;
  double bias;
  int n_inputs;
} perceptron;

perceptron *perceptron_alloc(const int n_inputs);
void        perceptron_free(perceptron *p);
void        perceptron_init(perceptron *p);
void        perceptron_forward(perceptron *p, const double *inputs, double (*activate)(double sum), double *output);
void        perceptron_backward(perceptron *p, const double *inputs, const double error, const double learning_rate);

#endif
