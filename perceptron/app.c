#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "perceptron.h"

#define N_INPUTS 5
#define N_ITERS 50
#define LR 0.1

void init(double *weights, int n_weights, double *bias);
double sigmoid(double sum);
void print_vec(double *vec, int n);

int main() {
  double inputs[] = {0.2, 0.5, 0.1, 0.3, 1.0};
  double weights[N_INPUTS];
  double output, t_output;
  double bias;
  double error;
  int i;

  srand(time(NULL));

  init(weights, N_INPUTS, &bias);

  t_output = 0.7;

  printf("===INIT===\n");
  printf("[INPUT] "); print_vec(inputs, N_INPUTS);
  printf("[WEIGHTS] "); print_vec(weights, N_INPUTS);
  printf("==========\n");

  for(i = 0; i < N_ITERS; i++) {
    perceptron_forward(inputs, weights, N_INPUTS, bias, sigmoid, &output);
    error = t_output - output;
    perceptron_backward(inputs, weights, N_INPUTS, &bias, error, LR);
    printf("===%dItrs===\n", i + 1);
    printf("[WEIGHTS] "); print_vec(weights, N_INPUTS);
    printf("[BIAS] "); printf("%4f\n", bias);
    printf("[OUTPUT] "); printf("%4f\n", output);
    printf("==========\n");
  }

  printf("output: %f\n", output);

  return 0;
}

void init(double *weights, int n_weights, double *bias) {
  int i;
  for(i = 0; i < n_weights; i++)
    weights[i] = (rand() % 100 - 50) / 50.0;
  *bias = (rand() % 101) / 100.0;
}

double sigmoid(double sum) {
  return 1.0 / (1.0 + exp(-sum));
}

void print_vec(double *vec, int n) {
  int i;
  for(i = 0; i < n; i++)
    printf("%.4f ", vec[i]);
  printf("\n");
}
