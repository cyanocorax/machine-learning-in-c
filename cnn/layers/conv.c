#include <stdlib.h>
#include "conv.h"

conv_layer *conv_layer_alloc(int in_ch, int out_ch, int kernel, int stride, int pad) {
  conv_layer *l = (conv_layer *)malloc(sizeof(conv_layer));
  l->weights = tensor_alloc(out_ch, in_ch, kernel, kernel);
  l->biases  = tensor_alloc(1, out_ch, 1, 1);
  l->stride  = stride;
  l->pad     = pad;
  return l;
}

void conv_layer_free(conv_layer *l) {
  if(!l) return;
  tensor_free(l->weights);
  tensor_free(l->biases);
  free(l);
}

static double dot_single_channel(const tensor *in, const tensor *w, int n, int oc, int ic, int oh, int ow, int stride, int pad) {
  double sum;
  int kh, kw, ih, iw;

  sum = 0.0;
  for(kh = 0; kh < w->h; kh++) {
    for(kw = 0; kw < w->w; kw++) {
      ih = oh * stride + kh - pad;
      iw = ow * stride + kw - pad;
      if(ih < 0 || ih >= in->h || iw < 0 || iw >= in->w) continue;
      sum += T_AT(in, n, ic, ih, iw) * T_AT(w, oc, ic, kh, kw);
    }
  }
  return sum;
}

static double conv_output_pixel(const conv_layer *l, const tensor *in, int n, int oc, int oh, int ow) {
  double val;
  int ic;

  val = l->biases->data[oc];
  for(ic = 0; ic < in->c; ic++)
    val += dot_single_channel(in, l->weights, n, oc, ic, oh, ow, l->stride, l->pad);
  return val;
}

void conv_layer_forward(const conv_layer *l, const tensor *in, tensor *out) {
  int n, oc, oh, ow;
  
  for(n = 0; n < in->n; n++)
    for(oc = 0; oc < out->c; oc++)
      for(oh = 0; oh < out->h; oh++)
        for(ow = 0; ow < out->w; ow++)
          T_AT(out, n, oc, oh, ow) = conv_output_pixel(l, in, n, oc, oh, ow);
}
