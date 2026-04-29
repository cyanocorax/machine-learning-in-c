#ifndef _ACTIVATIONS_
#define _ACTIVATIONS_

#include <math.h>

double relu(double x);
double relu_deriv(double x);

double sigmoid(double x);
double sigmoid_deriv(double x);

double tanh_fn(double x);
double tanh_deriv(double x);

typedef double (*activation_fn)(double);
typedef double (*activation_fn_deriv)(double);

typedef struct {
  activation_fn       forward;
  activation_fn_deriv backward;
} activation_def;

extern const activation_def ACT_RELU;
extern const activation_def ACT_SIGMOID;
extern const activation_def ACT_TANH;

#endif
