#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "perceptron.h"

#define N_SAMPLES 4
#define N_INPUTS 2
#define N_ITERS 50
#define LR 1.0

double sigmoid(double sum);
void print_vec(double *vec, int n);

int main() {
  perceptron *net;
  double inputs[N_SAMPLES][N_INPUTS] = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
  };
  double outputs[N_SAMPLES];
  double targets[] = {0.0, 0.0, 0.0, 1.0};
  double error;
  int i, j;

  srand(time(NULL));

  net = perceptron_alloc(N_INPUTS);

  perceptron_init(net);

  for(i = 0; i < N_ITERS; i++) {
    for(j = 0; j < N_SAMPLES; j++) {
      perceptron_forward(net, inputs[j], sigmoid, &outputs[j]);
      error = targets[j] - outputs[j];
      perceptron_backward(net, inputs[j], error, LR);
    }
    printf("Epoch %03d | error: %.3f | w=[%.3f, %.3f] b=%.3f\n", i + 1, error, net->weights[0], net->weights[1], net->bias);
  }

  printf("[Outputs] ");
  for(i = 0; i < N_SAMPLES; i++)
    printf("%.3f ", outputs[i]);
  printf("\n");

  perceptron_free(net);

  return 0;
}

double sigmoid(double sum) {
  return 1.0 / (1.0 + exp(-sum));
}
