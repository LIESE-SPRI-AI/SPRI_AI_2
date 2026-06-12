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




MobileDoubleConv2D* MobileDoubleConv2D_Create(int in_channels, int out_channels) {

    MobileDoubleConv2D* this = (MobileDoubleConv2D*)malloc(sizeof(MobileDoubleConv2D));

    if(this) {

        this->depthwise1 = Conv2D_Create(in_channels, in_channels, 3, 1, 1);
        this->bn1 = BatchNorm2D_Create(in_channels);

        this->pointwise1 = Conv2D_Create(in_channels, out_channels, 1, 1, 0);
        this->bn2 = BatchNorm2D_Create(out_channels);

        this->depthwise2 = Conv2D_Create(out_channels, out_channels, 3, 1, 1);
        this->bn3 = BatchNorm2D_Create(out_channels);

        this->pointwise2 = Conv2D_Create(out_channels, out_channels, 1, 1, 0);
        this->bn4 = BatchNorm2D_Create(out_channels);

        if(!this->depthwise1 || !this->bn1 || !this->pointwise1 || !this->bn2 || !this->depthwise2 || !this->bn3 || !this->pointwise2 || !this->bn4) {

            MobileDoubleConv2D_Destroy(this);

            return NULL;
        }
    }

    return this;
}


void MobileDoubleConv2D_Destroy(MobileDoubleConv2D* this) {

    if(this) {

        Conv2D_Destroy(this->depthwise1);
        BatchNorm2D_Destroy(this->bn1);

        Conv2D_Destroy(this->pointwise1);
        BatchNorm2D_Destroy(this->bn2);

        Conv2D_Destroy(this->depthwise2);
        BatchNorm2D_Destroy(this->bn3);

        Conv2D_Destroy(this->pointwise2);
        BatchNorm2D_Destroy(this->bn4);

        free(this);
    }
}


Tensor* MobileDoubleConv2D_Forward(MobileDoubleConv2D* this, Tensor* input) {

    Tensor* x = Conv2D_DepthwiseForward(this->depthwise1, input);
    BatchNorm2D_Forward(this->bn1, x);
    ReLU_Forward(x);

    Tensor* y = Conv2D_Forward(this->pointwise1, x);
    BatchNorm2D_Forward(this->bn2, y);
    ReLU_Forward(y);

    Tensor_Destroy(x);

    x = Conv2D_DepthwiseForward(this->depthwise2, y);
    BatchNorm2D_Forward(this->bn3, x);
    ReLU_Forward(x);

    Tensor_Destroy(y);

    y = Conv2D_Forward(this->pointwise2, x);
    BatchNorm2D_Forward(this->bn4, y);
    ReLU_Forward(y);

    Tensor_Destroy(x);

    return y;
}




