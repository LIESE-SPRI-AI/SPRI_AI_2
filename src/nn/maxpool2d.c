#include "nn/maxpool2d.h"

Tensor* MaxPool2D_Forward(Tensor* input) {

    int out_height = input->height / 2;
    int out_width  = input->width  / 2;

    Tensor* output = Tensor_Create(input->channels, out_height, out_width);

    if(!output) {
        return NULL;
    }

    int input_channel_size  = input->height * input->width;
    int output_channel_size = out_height * out_width;

    for(int c = 0; c < input->channels; c++) {

        type* input_ptr  = input->data  + c * input_channel_size;
        type* output_ptr = output->data + c * output_channel_size;

        for(int h = 0; h < out_height; h++) {

            type* row0 = input_ptr + (h * 2 + 0) * input->width;
            type* row1 = input_ptr + (h * 2 + 1) * input->width;

            for(int w = 0; w < out_width; w++) {

                int iw = w * 2;

                type v0 = row0[iw + 0];
                type v1 = row0[iw + 1];
                type v2 = row1[iw + 0];
                type v3 = row1[iw + 1];

                type max = v0;

                if(v1 > max) max = v1;
                if(v2 > max) max = v2;
                if(v3 > max) max = v3;

                output_ptr[h * out_width + w] = max;
            }
        }
    }

    return output;
}