#include "nn/conv2d.h"

Conv2D* Conv2D_Create(int in_channels,int out_channels,int kernel_size,int stride,int padding){

    Conv2D* this = (Conv2D*)malloc(sizeof(Conv2D));

    if(this){
        
        int weight_count =out_channels *in_channels *kernel_size *kernel_size;

        this->in_channels  = in_channels;
        this->out_channels = out_channels;

        this->kernel_size = kernel_size;

        this->stride  = stride;
        this->padding = padding;

        this->weights =(type*)malloc(sizeof(type) * weight_count);

        this->bias =(type*)malloc(sizeof(type) * out_channels);

        if(!this->weights || !this->bias){

            free(this->weights);
            free(this->bias);
            free(this);

            return NULL;
        }

        for(int i = 0; i < weight_count; i++){
            this->weights[i] = 0.0f;
        }

        for(int i = 0; i < out_channels; i++){
            this->bias[i] = 0.0f;
        }

    }
    return this;

}

void Conv2D_Destroy(Conv2D* this)
{
    if(this)
    {
        free(this->weights);
        free(this->bias);
        free(this);
    }
}


Tensor* Conv2D_Forward(Conv2D* this, Tensor* input)
{
    int out_height = (input->height + 2 * this->padding - this->kernel_size) / this->stride + 1;
    int out_width  = (input->width  + 2 * this->padding - this->kernel_size) / this->stride + 1;

    Tensor* output = Tensor_Create(this->out_channels, out_height, out_width);

    if(!output){

        return NULL;
    }

    int input_channel_size  = input->height * input->width;
    int output_channel_size = out_height * out_width;
    int kernel_channel_size = this->kernel_size * this->kernel_size;

    //#pragma omp parallel for
    for(int oc = 0; oc < this->out_channels; oc++)
    {
        type* output_ptr = output->data + oc * output_channel_size;

        for(int oh = 0; oh < out_height; oh++)
        {
            for(int ow = 0; ow < out_width; ow++)
            {
                type sum = this->bias[oc];

                for(int ic = 0; ic < this->in_channels; ic++)
                {
                    type* input_ptr  = input->data + ic * input_channel_size;
                    type* weight_ptr = this->weights + (oc * this->in_channels + ic) * kernel_channel_size;

                    for(int kh = 0; kh < this->kernel_size; kh++)
                    {
                        int ih = oh * this->stride + kh - this->padding;

                        if(ih < 0 || ih >= input->height){
                            continue;
                        }

                        int input_row  = ih * input->width;
                        int weight_row = kh * this->kernel_size;

                        for(int kw = 0; kw < this->kernel_size; kw++){

                            int iw = ow * this->stride + kw - this->padding;

                            if(iw < 0 || iw >= input->width){
                                continue;
                            }

                            sum += input_ptr[input_row + iw] * weight_ptr[weight_row + kw];
                        }
                    }
                }
                output_ptr[oh * out_width + ow] = sum;
            }
        }
    }

    return output;
}


// Tensor* Conv2D_Forward(Conv2D* this, Tensor* input)
// {
//     int out_height = input->height;
//     int out_width  = input->width;

//     Tensor* output = Tensor_Create(this->out_channels, out_height, out_width);

//     if(!output){
//         return NULL;
//     }

//     int input_channel_size  = input->height * input->width;
//     int output_channel_size = out_height * out_width;

//     #pragma omp parallel for
//     for(int oc = 0; oc < this->out_channels; oc++)
//     {
//         type* output_ptr = output->data + oc * output_channel_size;

//         for(int oh = 0; oh < out_height; oh++)
//         {
//             for(int ow = 0; ow < out_width; ow++)
//             {
//                 type sum = this->bias[oc];

//                 for(int ic = 0; ic < this->in_channels; ic++)
//                 {
//                     type* input_ptr  = input->data + ic * input_channel_size;
//                     type* weight_ptr = this->weights + (oc * this->in_channels + ic) * 9;

//                     int ih0 = oh - 1;
//                     int ih1 = oh;
//                     int ih2 = oh + 1;

//                     int iw0 = ow - 1;
//                     int iw1 = ow;
//                     int iw2 = ow + 1;

//                     if(ih0 >= 0)
//                     {
//                         int row = ih0 * input->width;

//                         if(iw0 >= 0)
//                             sum += input_ptr[row + iw0] * weight_ptr[0];
//                         sum += input_ptr[row + iw1] * weight_ptr[1];
//                         if(iw2 < input->width)   
//                             sum += input_ptr[row + iw2] * weight_ptr[2];

//                     }

//                     {
//                         int row = ih1 * input->width;

//                         if(iw0 >= 0)
//                             sum += input_ptr[row + iw0] * weight_ptr[3];
//                         sum += input_ptr[row + iw1] * weight_ptr[4];
//                         if(iw2 < input->width)   
//                             sum += input_ptr[row + iw2] * weight_ptr[5];
//                     }

//                     if(ih2 < input->height)
//                     {
//                         int row = ih2 * input->width;

//                         if(iw0 >= 0)
//                             sum += input_ptr[row + iw0] * weight_ptr[6];
//                         sum += input_ptr[row + iw1] * weight_ptr[7];
//                         if(iw2 < input->width)   
//                             sum += input_ptr[row + iw2] * weight_ptr[8];
//                     }
//                 }

//                 output_ptr[oh * out_width + ow] = sum;
//             }
//         }
//     }

//     return output;
// }
