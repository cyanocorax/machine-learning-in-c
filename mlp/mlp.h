#ifndef _MLP_
#define _MLP_

#include "activations.h"

typedef struct {
  int     in_size;
  int     out_size;
  double *weights;     // [out_size * in_size]
  double *biases;      // [out_size]
  double *z;           // [out_size] linear transformed
  double *a;           // [out_size] activated
  double *input_cache; // [in_size] store inputs
  double *dw;          // [out_size * in_size]
  double *db;          // [out_size]
  double *delta;       // [out_size]
} mlp_layer;

typedef struct {
  int             n_layers;
  mlp_layer      *layers;
  activation_def *activations;
} mlp;

// setup
void mlp_init(mlp *net, const int *layer_sizes, int n_layers);
void mlp_free(mlp *net);
void mlp_set_activation(mlp *net, int layer_inx, const activation_def *activation);
// forward
void mlp_layer_forward(mlp_layer *l, const double *inputs, const activation_def *activation);
void mlp_forward(mlp *net, const double *inputs);
// backward
void mlp_layer_backward(mlp_layer *l, const double *next_delta, const double *next_weights, int next_out_size, const activation_def *activation);
void mlp_backward(mlp *net, const double *targets);
// update
void mlp_update(mlp *net, double lr);

#endif
