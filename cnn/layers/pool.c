#include <float.h>
#include "pool.h"

static double max_in_window(const tensor *in, int n, int c, int oh, int ow, int kernel, int stride) {
  double max, v;
  int kh, kw, ih, iw;

  max = -DBL_MAX;
  for(kh = 0; kh < kernel; kh++) {
    for(kw = 0; kw < kernel; kw++) {
      ih = oh * stride + kh;
      iw = ow * stride + kw;
      if(ih >= in->h || iw >= in->w) continue;
      v = T_AT(in, n, c, ih, iw);
      if(v > max) max = v;
    }
  }
  return max;
}

void maxpool_forward(const tensor *in, tensor *out, int kernel, int stride) {
  int n, c, oh, ow;

  for(n = 0; n < in->n; n++)
    for(c = 0; c < in->c; c++)
      for(oh = 0; oh < out->h; oh++)
        for(ow = 0; ow < out->w; ow++)
          T_AT(out, n, c, oh, ow) = max_in_window(in, n, c, oh, ow, kernel, stride);
}
