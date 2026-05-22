#include "nn/doubleconv2d.h"

DoubleConv2D* DoubleConv2D_Create(int in_channels, int out_channels){

    DoubleConv2D* this = (DoubleConv2D*)malloc(sizeof(DoubleConv2D));

    if(this){

        this->conv1 = Conv2D_Create(in_channels, out_channels, 3, 1, 1);
        this->bn1   = BatchNorm2D_Create(out_channels);

        this->conv2 = Conv2D_Create(out_channels, out_channels, 3, 1, 1);
        this->bn2   = BatchNorm2D_Create(out_channels);

        if(!this->conv1 || !this->bn1 || !this->conv2 || !this->bn2){
            DoubleConv2D_Destroy(this);

            return NULL;
        }
    }

    return this;

}

void DoubleConv2D_Destroy(DoubleConv2D* this)
{
    if(this){
        Conv2D_Destroy(this->conv1);
        BatchNorm2D_Destroy(this->bn1);

        Conv2D_Destroy(this->conv2);
        BatchNorm2D_Destroy(this->bn2);

        free(this);
    }
}

Tensor* DoubleConv2D_Forward(DoubleConv2D* this, Tensor* input){

    Tensor* x = Conv2D_Forward(this->conv1, input);
    BatchNorm2D_Forward(this->bn1, x);
    ReLU_Forward(x);

    Tensor* y = Conv2D_Forward(this->conv2, x);
    BatchNorm2D_Forward(this->bn2, y);
    ReLU_Forward(y);

    Tensor_Destroy(x);

    return y;
    
}


