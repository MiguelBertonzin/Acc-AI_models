#ifndef LENET_MNIST_CAP64_HLS_H_
#define LENET_MNIST_CAP64_HLS_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void lenet_mnist_cap64_hls(
    hls::stream<input_t> &input_layer,
    hls::stream<result_t> &layer13_out
);

// hls-fpga-machine-learning insert emulator-defines


#endif
