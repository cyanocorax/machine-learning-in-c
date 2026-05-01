#include "activation.h"

void relu_forward(tensor *t) {
  int size;
  int i;

  size = t->n * t->c * t->h * t->w;
  for(i = 0; i < size; i++)
    if(t->data[i] < 0.0) t->data[i] = 0.0;
}
