#ifndef _TENSOR_H_
#define _TENSOR_H_

typedef struct {
  double *data;
  int n, c, h, w;
} tensor;

#define T_AT(t, _n, _c, _h, _w) (t)->data[((_n)*(t)->c+(_c))*(t)->h*(t)->w+(_h)*(t)->w+(_w)]

tensor *tensor_alloc(int n, int c, int h, int w);
void    tensor_free(tensor *t);
void    tensor_fill(tensor *t, double val);

#endif
