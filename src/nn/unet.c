#include "nn/unet.h"

UNet* UNet_Create(int in_channels, int out_channels){

    UNet* this = (UNet*)malloc(sizeof(UNet));

    if(this){

        this->enc1 = DoubleConv2D_Create(in_channels, 64);
        this->enc2 = DoubleConv2D_Create(64, 128);
        this->enc3 = DoubleConv2D_Create(128, 256);
        this->enc4 = DoubleConv2D_Create(256, 512);

        this->bottleneck = DoubleConv2D_Create(512, 1024);

        this->upconv4 = ConvTranspose2D_Create(1024, 512);
        this->dec4    = DoubleConv2D_Create(1024, 512);

        this->upconv3 = ConvTranspose2D_Create(512, 256);
        this->dec3    = DoubleConv2D_Create(512, 256);

        this->upconv2 = ConvTranspose2D_Create(256, 128);
        this->dec2    = DoubleConv2D_Create(256, 128);

        this->upconv1 = ConvTranspose2D_Create(128, 64);
        this->dec1    = DoubleConv2D_Create(128, 64);

        this->out_conv = Conv2D_Create(64, out_channels, 1, 1, 0);

        if(
            !this->enc1 ||
            !this->enc2 ||
            !this->enc3 ||
            !this->enc4 ||
            !this->bottleneck ||
            !this->upconv4 ||
            !this->dec4 ||
            !this->upconv3 ||
            !this->dec3 ||
            !this->upconv2 ||
            !this->dec2 ||
            !this->upconv1 ||
            !this->dec1 ||
            !this->out_conv
        ) {

            UNet_Destroy(this);
            return NULL;
        }
    }

    return this;
}

void UNet_Destroy(UNet* this) {

    if(this) {

        DoubleConv2D_Destroy(this->enc1);
        DoubleConv2D_Destroy(this->enc2);
        DoubleConv2D_Destroy(this->enc3);
        DoubleConv2D_Destroy(this->enc4);

        DoubleConv2D_Destroy(this->bottleneck);

        ConvTranspose2D_Destroy(this->upconv4);
        DoubleConv2D_Destroy(this->dec4);

        ConvTranspose2D_Destroy(this->upconv3);
        DoubleConv2D_Destroy(this->dec3);

        ConvTranspose2D_Destroy(this->upconv2);
        DoubleConv2D_Destroy(this->dec2);

        ConvTranspose2D_Destroy(this->upconv1);
        DoubleConv2D_Destroy(this->dec1);

        Conv2D_Destroy(this->out_conv);

        free(this);
    }
}


Tensor* UNet_Forward(UNet* this, Tensor* input) {

    //Encoder 1
    //printf("Encoder 1 \n");
    Tensor* e1 = DoubleConv2D_Forward(this->enc1, input);
    Tensor* p1 = MaxPool2D_Forward(e1);

    //Encoder 2
    //printf("Encoder 2 \n");
    Tensor* e2 = DoubleConv2D_Forward(this->enc2, p1);
    Tensor_Destroy(p1);
    Tensor* p2 = MaxPool2D_Forward(e2);

    //Encoder 3
    //printf("Encoder 3 \n");
    Tensor* e3 = DoubleConv2D_Forward(this->enc3, p2);
    Tensor_Destroy(p2);
    Tensor* p3 = MaxPool2D_Forward(e3);

    //Encoder 4
    //printf("Encoder 4 \n");
    Tensor* e4 = DoubleConv2D_Forward(this->enc4, p3);
    Tensor_Destroy(p3);
    Tensor* p4 = MaxPool2D_Forward(e4);

    //Bottleneck
    //printf("Bottleneck \n");
    Tensor* bottleneck =DoubleConv2D_Forward(this->bottleneck,p4);
    Tensor_Destroy(p4);

    //Decoder 4
    //printf("Decoder 4 \n");
    Tensor* d4 =ConvTranspose2D_Forward(this->upconv4,bottleneck);
    Tensor_Destroy(bottleneck);

    FILE * f = fopen("bottleneck_c.bin", "wb");

    fwrite(
        d4->data,
        sizeof(type),
        d4->channels * d4->height * d4->width,
        f
    );

    fclose(f);

    Tensor* c4 = Tensor_Concat(e4, d4);
    Tensor_Destroy(e4);
    Tensor_Destroy(d4);

    Tensor* y4 =DoubleConv2D_Forward(this->dec4,c4);
    Tensor_Destroy(c4);

    //Decoder 3
    //printf("Decoder 3 \n");
    Tensor* d3 =ConvTranspose2D_Forward( this->upconv3, y4);
    Tensor_Destroy(y4);

    Tensor* c3 = Tensor_Concat(e3, d3);
    Tensor_Destroy(e3);
    Tensor_Destroy(d3);

    Tensor* y3 =DoubleConv2D_Forward(this->dec3,c3);
    Tensor_Destroy(c3);

    //Decoder 2
    //printf("Decoder 2 \n");
    Tensor* d2 =ConvTranspose2D_Forward(this->upconv2, y3);
    Tensor_Destroy(y3);

    Tensor* c2 = Tensor_Concat(e2, d2);
    Tensor_Destroy(e2);
    Tensor_Destroy(d2);

    Tensor* y2 =DoubleConv2D_Forward(this->dec2,c2);
    Tensor_Destroy(c2);

    //Decoder 1
    //printf("Decoder 1 \n");
    Tensor* d1 =ConvTranspose2D_Forward(this->upconv1,y2);
    Tensor_Destroy(y2);

    Tensor* c1 = Tensor_Concat(e1, d1);
    Tensor_Destroy(e1);
    Tensor_Destroy(d1);

    Tensor* y1 =DoubleConv2D_Forward(this->dec1,c1);
    Tensor_Destroy(c1);

    //Output
    Tensor* output =Conv2D_Forward(this->out_conv,y1);
    Tensor_Destroy(y1);

    return output;
}



MobileUNet *MobileUNet_Create(int in_channels, int out_channels)
{

    MobileUNet *this = (MobileUNet *)malloc(sizeof(MobileUNet));

    if (this)
    {

        this->enc1 = MobileDoubleConv2D_Create(in_channels, 64);
        this->enc2 = MobileDoubleConv2D_Create(64, 128);
        this->enc3 = MobileDoubleConv2D_Create(128, 256);
        this->enc4 = MobileDoubleConv2D_Create(256, 512);

        this->bottleneck = MobileDoubleConv2D_Create(512, 1024);

        this->upconv4 = ConvTranspose2D_Create(1024, 512);
        this->dec4 = MobileDoubleConv2D_Create(1024, 512);

        this->upconv3 = ConvTranspose2D_Create(512, 256);
        this->dec3 = MobileDoubleConv2D_Create(512, 256);

        this->upconv2 = ConvTranspose2D_Create(256, 128);
        this->dec2 = MobileDoubleConv2D_Create(256, 128);

        this->upconv1 = ConvTranspose2D_Create(128, 64);
        this->dec1 = MobileDoubleConv2D_Create(128, 64);

        this->out_conv = Conv2D_Create(64, out_channels, 1, 1, 0);

        if (
            !this->enc1 ||
            !this->enc2 ||
            !this->enc3 ||
            !this->enc4 ||
            !this->bottleneck ||
            !this->upconv4 ||
            !this->dec4 ||
            !this->upconv3 ||
            !this->dec3 ||
            !this->upconv2 ||
            !this->dec2 ||
            !this->upconv1 ||
            !this->dec1 ||
            !this->out_conv)
        {

            MobileUNet_Destroy(this);
            return NULL;
        }
    }

    return this;
}

void MobileUNet_Destroy(MobileUNet *this)
{

    if (this)
    {

        MobileDoubleConv2D_Destroy(this->enc1);
        MobileDoubleConv2D_Destroy(this->enc2);
        MobileDoubleConv2D_Destroy(this->enc3);
        MobileDoubleConv2D_Destroy(this->enc4);

        MobileDoubleConv2D_Destroy(this->bottleneck);

        ConvTranspose2D_Destroy(this->upconv4);
        MobileDoubleConv2D_Destroy(this->dec4);

        ConvTranspose2D_Destroy(this->upconv3);
        MobileDoubleConv2D_Destroy(this->dec3);

        ConvTranspose2D_Destroy(this->upconv2);
        MobileDoubleConv2D_Destroy(this->dec2);

        ConvTranspose2D_Destroy(this->upconv1);
        MobileDoubleConv2D_Destroy(this->dec1);

        Conv2D_Destroy(this->out_conv);

        free(this);
    }
}



Tensor* MobileUNet_Forward(MobileUNet* this, Tensor* input) {

    Tensor* e1 = MobileDoubleConv2D_Forward(this->enc1, input);
    Tensor* p1 = MaxPool2D_Forward(e1);

    Tensor* e2 = MobileDoubleConv2D_Forward(this->enc2, p1);
    Tensor_Destroy(p1);
    Tensor* p2 = MaxPool2D_Forward(e2);

    Tensor* e3 = MobileDoubleConv2D_Forward(this->enc3, p2);
    Tensor_Destroy(p2);
    Tensor* p3 = MaxPool2D_Forward(e3);

    Tensor* e4 = MobileDoubleConv2D_Forward(this->enc4, p3);
    Tensor_Destroy(p3);
    Tensor* p4 = MaxPool2D_Forward(e4);

    Tensor* bottleneck = MobileDoubleConv2D_Forward(this->bottleneck, p4);
    Tensor_Destroy(p4);

    Tensor* d4 = ConvTranspose2D_Forward(this->upconv4, bottleneck);
    Tensor_Destroy(bottleneck);
    Tensor* c4 = Tensor_Concat(e4, d4);
    Tensor_Destroy(e4);
    Tensor_Destroy(d4);
    Tensor* y4 = MobileDoubleConv2D_Forward(this->dec4, c4);
    Tensor_Destroy(c4);

    Tensor* d3 = ConvTranspose2D_Forward(this->upconv3, y4);
    Tensor_Destroy(y4);
    Tensor* c3 = Tensor_Concat(e3, d3);
    Tensor_Destroy(e3);
    Tensor_Destroy(d3);
    Tensor* y3 = MobileDoubleConv2D_Forward(this->dec3, c3);
    Tensor_Destroy(c3);

    Tensor* d2 = ConvTranspose2D_Forward(this->upconv2, y3);
    Tensor_Destroy(y3);
    Tensor* c2 = Tensor_Concat(e2, d2);
    Tensor_Destroy(e2);
    Tensor_Destroy(d2);
    Tensor* y2 = MobileDoubleConv2D_Forward(this->dec2, c2);
    Tensor_Destroy(c2);

    Tensor* d1 = ConvTranspose2D_Forward(this->upconv1, y2);
    Tensor_Destroy(y2);
    Tensor* c1 = Tensor_Concat(e1, d1);
    Tensor_Destroy(e1);
    Tensor_Destroy(d1);
    Tensor* y1 = MobileDoubleConv2D_Forward(this->dec1, c1);
    Tensor_Destroy(c1);

    Tensor* output = Conv2D_Forward(this->out_conv, y1);
    Tensor_Destroy(y1);

    return output;
}


