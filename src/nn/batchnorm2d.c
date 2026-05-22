#include "nn/batchnorm2d.h"


BatchNorm2D* BatchNorm2D_Create(int channels)
{
    BatchNorm2D* this = (BatchNorm2D*)malloc(sizeof(BatchNorm2D));

    if(this)
    {
        this->channels = channels;

        this->epsilon = 1e-5f;

        this->weight = (type*)malloc(sizeof(type) * channels);
        this->bias = (type*)malloc(sizeof(type) * channels);

        this->running_mean = (type*)malloc(sizeof(type) * channels);
        this->running_variance = (type*)malloc(sizeof(type) * channels);

        if(!this->weight || !this->bias || !this->running_mean || !this->running_variance){

            free(this->weight);
            free(this->bias);

            free(this->running_mean);
            free(this->running_variance);

            free(this);

            return NULL;
        }

        for(int c = 0; c < channels; c++){

            this->weight[c] = 1.0f;
            this->bias[c] = 0.0f;

            this->running_mean[c] = 0.0f;
            this->running_variance[c] = 1.0f;
        }
    }

    return this;
}

void BatchNorm2D_Destroy(BatchNorm2D* this)
{
    if(this)
    {
        free(this->weight);
        free(this->bias);

        free(this->running_mean);
        free(this->running_variance);

        free(this);
    }
}


void BatchNorm2D_Forward(BatchNorm2D* this, Tensor* x)
{
    int channel_size = x->height * x->width;

    for(int c = 0; c < x->channels; c++)
    {
        type weight = this->weight[c];
        type bias = this->bias[c];

        type mean = this->running_mean[c];
        type variance = this->running_variance[c];

        type inv_std = 1.0f / sqrtf(variance + this->epsilon);

        type* current = x->data + c * channel_size;
        type* end = current + channel_size;

        while(current < end){

            *current = weight * ((*current - mean) * inv_std) + bias;

            current++;
        }
    }
}



// void BatchNorm2D_Forward(BatchNorm2D* this, Tensor* x)
// {
//     int channel_size =x->height * x->width;

//     for(int c = 0; c < x->channels; c++)
//     {
//         type mean = 0.0f;

//         type* ptr =x->data + c * channel_size;

//         type* end = ptr + channel_size;

//         type* current = ptr;

//         while(current < end){

//             mean += *current;
//             current++;
//         }

//         mean /= channel_size;

//         type variance = 0.0f;

//         current = ptr;

//         while(current < end){

//             type diff = *current - mean;

//             variance += diff * diff;

//             current++;
//         }

//         variance /= channel_size;

//         type inv_std = 1.0f / sqrtf( variance + this->epsilon );

//         current = ptr;

//         while(current < end){

//             *current =this->gamma[c] *((*current - mean) * inv_std)+this->beta[c];
//             current++;

//         }
//     }
// }