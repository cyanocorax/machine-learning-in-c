#include <stdlib.h>
#include "fc.h"

fc_layer *fc_layer_alloc(int in_size, int out_size) {
  fc_layer *l = (fc_layer *)malloc(sizeof(fc_layer));
  l->weights = tensor_alloc(out_size, in_size, 1, 1);
  l->biases  = tensor_alloc(out_size, 1, 1, 1);
  return l;
}

void fc_layer_free(fc_layer *l) {
  tensor_free(l->weights);
  tensor_free(l->biases);
  free(l);
}

static double dot_row(const double *w_row, const double *x, int len) {
  double sum;
  int i;

  sum = 0.0;
  for(i = 0; i < len; i++) sum += w_row[i] * x[i];
  return sum;
}

void fc_layer_forward(const fc_layer *l, const tensor *in, tensor *out) {
  int in_size, out_size;
  double *y;
  int n, o;

  in_size = in->c * in->h * in->w;
  out_size = out->c * out->h * out->w;
  for(n = 0; n < in->n; n++) {
    const double *x = in->data + n * in_size;
    y = out->data + n * out_size;
    for(o = 0; o < out_size; o++)
      y[o] = l->biases->data[o] + dot_row(l->weights->data + o * in_size, x, in_size);
  }
}
