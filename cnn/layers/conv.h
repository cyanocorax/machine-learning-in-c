#ifndef _CONV_H_
#define _CONV_H_

#include "tensor.h"

typedef struct {
  tensor *weights; // (out_ch, in_ch, kH, kW)
  tensor *biases;  // (1, out_ch, 1, 1)
  int stride;
  int pad;
} conv_layer;

conv_layer *conv_layer_alloc(int in_ch, int out_ch, int kernel, int stride, int pad);
void        conv_layer_free(conv_layer *l);
void        conv_layer_forward(const conv_layer *l, const tensor *in, tensor *out);

#endif
