
#include "nn/tensor.h"

Tensor * Tensor_Create(int c, int h, int w){

    Tensor * this = (Tensor*)malloc(sizeof(Tensor));

    if(!this)
    {
        free(this);
        return NULL;
    }
    this->data = (type*)malloc(sizeof(type) * c * h * w);

    if(!this->data)
    {
        free(this);
        return NULL;
    }

    this->channels = c;
    this->height   = h;
    this->width    = w;

    this->size = c * h * w;

    return this;
}

void Tensor_Destroy(Tensor* this)
{
    if(this)
    {
        free(this->data);
        free(this);
    }
}

type Tensor_Get(Tensor* this,int c,int h,int w)
{
    int idx = c * this->height * this->width + h * this->width + w;

    return this->data[idx];
}

void Tensor_Set(Tensor* this,int c,int h,int w,type value)
{
    int idx = c * this->height * this->width + h * this->width + w;

    this->data[idx] = value;

}

void Tensor_Print(Tensor * this){

    for(int c = 0; c < this->channels; c++){

        printf("Channel %d: \n",c);

        for(int h = 0; h < this->height; h++)
            {
                for(int w = 0; w < this->width; w++)
                {
                    printf("%8.8f,", Tensor_Get(this, c, h, w));
                }

                printf("\n");
            }

            printf("\n \n");
    }
}


Tensor* Tensor_Concat(Tensor* a, Tensor* b) {

    if(a->height != b->height || a->width  != b->width) {
        return NULL;
    }

    Tensor* output = Tensor_Create( a->channels + b->channels, a->height,a->width);

    if(!output) {
        return NULL;
    }

    type* dst = output->data;

    type* src = a->data;
    type* end = src + a->size;

    while(src < end) {
        *dst++ = *src++;
    }

    src = b->data;
    end = src + b->size;

    while(src < end) {
        *dst++ = *src++;
    }

    return output;
}
