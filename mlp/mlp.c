#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mlp.h"

void mlp_init(mlp *net, const int *layer_sizes, int n_layers) {
  int in, out;
  mlp_layer *l;
  double scale;
  int i, j;

  srand(time(NULL));

  net->n_layers    = n_layers - 1;
  net->layers      = malloc(sizeof(mlp_layer) * net->n_layers);
  net->activations = malloc(sizeof(activation_def) * net->n_layers);

  for(i = 0; i < net->n_layers; i++) {
    in  = layer_sizes[i];
    out = layer_sizes[i + 1];
    l   = &net->layers[i];

    l->in_size     = in;
    l->out_size    = out;
    l->weights     = malloc(sizeof(double) * in * out);
    l->biases      = malloc(sizeof(double) * out);
    l->z           = malloc(sizeof(double) * out);
    l->a           = malloc(sizeof(double) * out);
    l->input_cache = malloc(sizeof(double) * in);
    l->dw          = malloc(sizeof(double) * in * out);
    l->db          = malloc(sizeof(double) * out);
    l->delta       = malloc(sizeof(double) * out);

    // init He
    scale = sqrt(2.0 / in);
    for(j = 0; j < in * out; j++)
      l->weights[j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
  }
}

void mlp_free(mlp *net) {
  mlp_layer *l;
  int i;

  for(i = 0; i < net->n_layers; i++) {
    l = &net->layers[i];
    free(l->weights);
    free(l->biases);
    free(l->z);
    free(l->a);
    free(l->input_cache);
    free(l->dw);
    free(l->db);
    free(l->delta);
  }
  free(net->layers);
  free(net->activations);
  net->layers = NULL;
  net->activations = NULL;
  net->n_layers = 0;
}

void mlp_set_activation(mlp *net, int layer_inx, const activation_def *activation) {
  net->activations[layer_inx] = *activation;
}

void mlp_layer_forward(mlp_layer *l, const double *inputs, const activation_def *activation) {
  int i, j;

  memcpy(l->input_cache, inputs, l->in_size * sizeof(double));
  for(i = 0; i < l->out_size; i++) {
    l->z[i] = l->biases[i];
    for(j = 0; j < l->in_size; j++)
      l->z[i] += l->weights[i * l->in_size + j] * inputs[j];
    l->a[i] = activation->forward(l->z[i]);
  }
}

void mlp_forward(mlp *net, const double *inputs) {
  const double *in = inputs;
  int i;
  
  for(i = 0; i < net->n_layers; i++) {
    mlp_layer_forward(&net->layers[i], in, &net->activations[i]);
    in = net->layers[i].a;
  }
}

void mlp_layer_backward(mlp_layer *l, const double *next_delta, const double *next_weights, int next_out_size, const activation_def *activation) {
  double grad;
  int i, j, k;

  for(i = 0; i < l->out_size; i++) {
    grad = 0.0;
    for(k = 0; k < next_out_size; k++)
      grad += next_weights[k * l->out_size + i] * next_delta[k];
    l->delta[i] = grad * activation->backward(l->z[i]);
  }

  for(int i = 0; i < l->out_size; i++) {
    for(j = 0; j < l->in_size; j++)
      l->dw[i * l->in_size + j] = l->delta[i] * l->input_cache[j];
    l->db[i] = l->delta[i];
  }
}

void mlp_backward(mlp *net, const double *targets) {
  int last       = net->n_layers - 1;
  mlp_layer *out = &net->layers[last];
  int i, j;

  // output layer
  for(i = 0; i < out->out_size; i++)
    out->delta[i] = (out->a[i] - targets[i]) * net->activations[last].backward(out->z[i]);

  for(i = 0; i < out->out_size; i++) {
    for(j = 0; j < out->in_size; j++)
      out->dw[i * out->in_size + j] = out->delta[i] * out->input_cache[j];
    out->db[i] = out->delta[i];
  }

  // hidden layers
  for(i = last - 1; i >= 0; i--) {
    mlp_layer_backward(&net->layers[i], net->layers[i + 1].delta, net->layers[i + 1].weights, net->layers[i + 1].out_size, &net->activations[i]);
  }
}

// sgd
void mlp_update(mlp *net, double lr) {
  mlp_layer *l;
  int i, j;

  for(i = 0; i < net->n_layers; i++) {
    l = &net->layers[i];
    for(j = 0; j < l->out_size * l->in_size; j++)
      l->weights[j] -= lr * l->dw[j];
    for(j = 0; j < l->out_size; j++)
      l->biases[j] -= lr * l->db[j];
  }
}
