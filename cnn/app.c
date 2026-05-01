#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "layers/conv.h"
#include "layers/activation.h"
#include "layers/pool.h"
#include "layers/fc.h"

static void print_tensor(const char *label, const tensor *t) {
  int n, c, h, w;

  printf("=== %s (n=%d c=%d h=%d w=%d) ===\n", label, t->n, t->c, t->h, t->w);
  for(n = 0; n < t->n; n++)
    for(c = 0; c < t->c; c++) {
      printf("[n=%d c=%d]\n", n, c);
      for(h = 0; h < t->h; h++) {
        for(w = 0; w < t->w; w++)
          printf("%6.2f ", T_AT(t, n, c, h, w));
        printf("\n");
      }
    }
  printf("\n");
}

static void fill_input(tensor *t) {
  double vals[] = {
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16
  };
  int i;

  for(i = 0; i < 16; i++)
    t->data[i] = vals[i];
}

static void fill_weights(tensor *t) {
  double vals[] = {
    -1, -1, -1,
    -1,  0, -1,
    -1, -1, -1
  };
  int i;

  for(i = 0; i < 9; i++)
    t->data[i] = vals[i];
}

int main(void) {
  tensor *input;
  conv_layer *conv;
  tensor *conv_out;
  conv_layer *conv_pad;
  tensor *conv_pad_out;
  tensor *pool_out;
  fc_layer *fc;
  tensor *fc_in, *fc_out;
  int o, i;

  input = tensor_alloc(1, 1, 4, 4);
  fill_input(input);
  print_tensor("Input", input);

  conv = conv_layer_alloc(1, 1, 3, 1, 0);
  fill_weights(conv->weights);
  tensor_fill(conv->biases, 0.0);

  conv_out = tensor_alloc(1, 1, 2, 2);
  conv_layer_forward(conv, input, conv_out);
  print_tensor("After Conv (3x3, stride=1, pad=0)", conv_out);

  relu_forward(conv_out);
  print_tensor("After ReLU", conv_out);

  conv_pad = conv_layer_alloc(1, 1, 3, 1, 1);
  fill_weights(conv_pad->weights);
  tensor_fill(conv_pad->biases, 0.5);

  conv_pad_out = tensor_alloc(1, 1, 4, 4);
  conv_layer_forward(conv_pad, input, conv_pad_out);
  print_tensor("After Conv (pad=1, same size, bias=0.5)", conv_pad_out);

  pool_out = tensor_alloc(1, 1, 2, 2);
  maxpool_forward(conv_pad_out, pool_out, 2, 2);
  print_tensor("After MaxPool (2x2, stride=2)", pool_out);

  fc = fc_layer_alloc(4, 3);
  for(o = 0; o < 3; o++)
    for(i = 0; i < 4; i++)
      fc->weights->data[o * 4 + i] = (double)(o + i + 1) * 0.1;
  tensor_fill(fc->biases, 0.0);

  fc_in = tensor_alloc(1, 4, 1, 1);
  fc_out = tensor_alloc(1, 3, 1, 1);
  for(i = 0; i < 4; i++) fc_in->data[i] = pool_out->data[i];

  fc_layer_forward(fc, fc_in, fc_out);
  print_tensor("After FC (4->3)", fc_out);

  tensor_free(input);
  tensor_free(conv_out);
  tensor_free(conv_pad_out);
  tensor_free(pool_out);
  tensor_free(fc_in);
  tensor_free(fc_out);
  conv_layer_free(conv);
  conv_layer_free(conv_pad);
  fc_layer_free(fc);

  return 0;
}
