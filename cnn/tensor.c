#include <stdlib.h>
#include <string.h>
#include "tensor.h"

tensor *tensor_alloc(int n, int c, int h, int w) {
  tensor *t = (tensor *)malloc(sizeof(tensor));
  t->n = n; t->c = c; t->h = h; t->w = w;
  t->data = (double *)malloc(sizeof(double) * n * c * h * w);
  return t;
}

void tensor_free(tensor *t) {
  if(!t) return;
  free(t->data);
  free(t);
}

void tensor_fill(tensor *t, double val) {
  int size = t->n * t->c * t->h * t->w;
  int i;
  for(i = 0; i < size; i++) t->data[i] = val;
}
