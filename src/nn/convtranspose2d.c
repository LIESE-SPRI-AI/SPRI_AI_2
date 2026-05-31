
#include "nn\convtranspose2d.h"

ConvTranspose2D* ConvTranspose2D_Create(int in_channels, int out_channels) {

    ConvTranspose2D* this = (ConvTranspose2D*)malloc(sizeof(ConvTranspose2D));

    if(this) {

        int weight_count =out_channels * in_channels * 2 * 2;

        this->in_channels  = in_channels;
        this->out_channels = out_channels;

        this->weights = (type*)malloc(sizeof(type) * weight_count);

        this->bias =(type*)malloc(sizeof(type) * out_channels);

        if(!this->weights || !this->bias) {

            free(this->weights);
            free(this->bias);
            free(this);
            return NULL;

        }

        for(int i = 0; i < weight_count; i++) {
            this->weights[i] = 1.0f;
        }

        for(int i = 0; i < out_channels; i++) {
            this->bias[i] = 0.0f;
        }
    }

    return this;
}


void ConvTranspose2D_Destroy(ConvTranspose2D* this) {

    if(this) {

        free(this->weights);
        free(this->bias);

        free(this);
    }
}

Tensor* ConvTranspose2D_Forward(ConvTranspose2D* this, Tensor* input) {

    int out_height = input->height * 2;
    int out_width  = input->width  * 2;

    Tensor* output =
        Tensor_Create(this->out_channels,out_height,out_width);

    if(!output) {
        return NULL;
    }

    int input_channel_size  = input->height * input->width;
    int output_channel_size = out_height * out_width;

    for(int oc = 0; oc < this->out_channels; oc++) {

        type* output_ptr =output->data + oc * output_channel_size;

        for(int i = 0; i < output_channel_size; i++) {
            output_ptr[i] = this->bias[oc];
        }

        for(int ic = 0; ic < this->in_channels; ic++) {

            type* input_ptr = input->data + ic * input_channel_size;
            type* weight_ptr = this->weights + (ic * this->out_channels + oc) * 4;

            for(int ih = 0; ih < input->height; ih++) {
                
                for(int iw = 0; iw < input->width; iw++) {

                    type value = input_ptr[ih * input->width + iw];

                    int oh = ih * 2;
                    int ow = iw * 2;

                    output_ptr[(oh + 0) * out_width + (ow + 0)] += value * weight_ptr[0];
                    output_ptr[(oh + 0) * out_width + (ow + 1)] += value * weight_ptr[1];
                    output_ptr[(oh + 1) * out_width + (ow + 0)] += value * weight_ptr[2];
                    output_ptr[(oh + 1) * out_width + (ow + 1)] += value * weight_ptr[3];

                }
            }
        }
    }

    return output;
}


