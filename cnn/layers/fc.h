#ifndef _FC_H_
#define _FC_H_

#include "tensor.h"

typedef struct {
  tensor *weights; // (out, in, 1, 1)
  tensor *biases;  // (out)
} fc_layer;

fc_layer *fc_layer_alloc(int in_size, int out_size);
void      fc_layer_free(fc_layer *l);
void      fc_layer_forward(const fc_layer *l, const tensor *in, tensor *out);

#endif
