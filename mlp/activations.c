#include "activations.h"

const activation_def ACT_RELU    = { relu,    relu_deriv    };
const activation_def ACT_SIGMOID = { sigmoid, sigmoid_deriv };
const activation_def ACT_TANH    = { tanh_fn, tanh_deriv    };

double relu(double x) { return x > 0.0 ? x : 0.0; }
double relu_deriv(double x) { return x > 0.0 ? 1.0 : 0.0; }

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_deriv(double x) { double s = sigmoid(x); return s * (1.0 - s); }

double tanh_fn(double x) { return tanh(x); }
double tanh_deriv(double x) { double t = tanh_fn(x); return 1.0 - t * t; }
