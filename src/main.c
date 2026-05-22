#include "main.h"


int main()
{
    // Tensor de entrada 3x16x16
    Tensor* input = Tensor_Create(4, 16, 16);

    // Inicializar tensor de entrada (simulando la imagen de entrada)
    for(int i = 0; i < input->size; i++)
        input->data[i] = 1.0f;


    //4 canales de entrada, 4 canales de salida
    DoubleConv2D* model = DoubleConv2D_Create(4, 4);

    // Inicializar pesos y bias (sólo para prueba, esto se carga de los parámetros)
    for(int i = 0; i < model->conv1->out_channels * model->conv1->in_channels * model->conv1->kernel_size * model->conv1->kernel_size; i++)
        model->conv1->weights[i] = 0.01f;

    for(int i = 0; i < model->conv1->out_channels; i++)
        model->conv1->bias[i] = 0.0f;

    for(int i = 0; i < model->bn1->channels; i++)
    {
        model->bn1->weight[i] = 1.0f;
        model->bn1->bias[i] = 0.0f;
        model->bn1->running_mean[i] = 0.0f;
        model->bn1->running_variance[i] = 1.0f;
    }

    for(int i = 0; i < model->conv2->out_channels * model->conv2->in_channels * model->conv2->kernel_size * model->conv2->kernel_size; i++)
        model->conv2->weights[i] = 0.01f;

    for(int i = 0; i < model->conv2->out_channels; i++)
        model->conv2->bias[i] = 0.0f;

    for(int i = 0; i < model->bn2->channels; i++)
    {
        model->bn2->weight[i] = 1.0f;
        model->bn2->bias[i] = 0.0f;
        model->bn2->running_mean[i] = 0.0f;
        model->bn2->running_variance[i] = 1.0f;
    }


    // Aplicamos la doble convolución
    Tensor* output = DoubleConv2D_Forward(model, input);


    // Imprimimos el resultado
    Tensor_Print(output);

    // Liberamos memoria
    Tensor_Destroy(input);
    Tensor_Destroy(output);

    DoubleConv2D_Destroy(model);

    return 0;
}