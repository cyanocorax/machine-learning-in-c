#include <stdio.h>
#include <stdlib.h>
#include "mlp.h"

#define EPOCHS 100
#define LR 0.01

void print_vec(char *head, double *vec, int n_vec);

int main() {
  mlp net;
  int sizes[] = {784, 128, 64, 10};
  double inputs[784];
  double targets[10] = { 0 };
  targets[3] = 1.0;
  int i;

  mlp_init(&net, sizes, sizeof(sizes)/sizeof(sizes[0]));
  mlp_set_activation(&net, 0, &ACT_RELU);
  mlp_set_activation(&net, 1, &ACT_RELU);
  mlp_set_activation(&net, 2, &ACT_SIGMOID);

  for(i = 0; i < sizeof(inputs)/sizeof(inputs[0]); i++)
    inputs[i] = (rand() % 100) / 100.0;

  for(i = 0; i < EPOCHS; i++) {
    mlp_forward(&net, inputs);
    mlp_backward(&net, targets);
    mlp_update(&net, LR);
  }

  print_vec("[output]  ", net.layers[net.n_layers - 1].a, net.layers[net.n_layers - 1].out_size);
  print_vec("[targets] ", targets, 10);

  mlp_free(&net);

  return 0;
}

void print_vec(char *head, double *vec, int n_vec) {
  int i;
  
  printf("%s ", head);
  for(i = 0; i < n_vec; i++) printf("%.2f ", vec[i]);
  printf("\n");
}
