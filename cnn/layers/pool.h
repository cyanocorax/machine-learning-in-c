#ifndef _POOL_H_
#define _POOL_H_

#include "tensor.h"

void maxpool_forward(const tensor *in, tensor *out, int kernel, int stride);

#endif
