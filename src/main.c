#include "main.h"

#include <time.h>

double Timer_Now(void) { 

    return (double)clock() / CLOCKS_PER_SEC; 

}

int main() {

    Tensor* input = Tensor_Create(4, 16, 16);

    for(int c = 0; c < input->channels; c++) {
        for(int h = 0; h < input->height; h++) {
            for(int w = 0; w < input->width; w++) {

                int idx =
                    c * input->height * input->width +
                    h * input->width +
                    w;

                input->data[idx] =
                    (type)(
                        c +
                        h * 0.1f +
                        w * 0.01f
                    );
            }
        }
    }

    UNet* model = UNet_Create(4, 2);

    if(!model) {

        printf("Error\n");

        Tensor_Destroy(input);

        return -1;
    }

    printf("Forward\n");
    double start = Timer_Now();
    Tensor* output =UNet_Forward(model, input);
    double end = Timer_Now();
    double run_time_ms =(end - start);
    printf("Time: %.16f ms\n", run_time_ms);
    
    printf("Out Forward\n");
    if(!output) {

        printf("Error\n");

        UNet_Destroy(model);

        Tensor_Destroy(input);

        return -1;
    }

    printf(
        "Shape: %d x %d x %d\n\n",
        output->channels,
        output->height,
        output->width
    );

    Tensor_Print(output);

    Tensor_Destroy(output);
    UNet_Destroy(model);
    Tensor_Destroy(input);

    return 0;
}


// int main()
// {
//     // Tensor de entrada 4x16x16
//     Tensor* input = Tensor_Create(4, 16, 16);

//     // Inicializar tensor de entrada (simulando la imagen de entrada)
//     for(int i = 0; i < input->size; i++)
//         input->data[i] = 1.0f;


//     //4 canales de entrada, 4 canales de salida
//     DoubleConv2D* model = DoubleConv2D_Create(4, 4);

//     // ConvTranspose2D 4 canales de entrada, 4 canales de salida. Aumenta el tamaño de cada capa.
//     ConvTranspose2D* up = ConvTranspose2D_Create(4, 4);


//     // Inicializar pesos y bias (sólo para prueba, esto se carga de los parámetros)
//     for(int i = 0; i < model->conv1->out_channels * model->conv1->in_channels * model->conv1->kernel_size * model->conv1->kernel_size; i++)
//         model->conv1->weights[i] = 0.01f;

//     for(int i = 0; i < model->conv1->out_channels; i++)
//         model->conv1->bias[i] = 0.0f;

//     for(int i = 0; i < model->bn1->channels; i++)
//     {
//         model->bn1->weight[i] = 1.0f;
//         model->bn1->bias[i] = 0.0f;
//         model->bn1->running_mean[i] = 0.0f;
//         model->bn1->running_variance[i] = 1.0f;
//     }

//     for(int i = 0; i < model->conv2->out_channels * model->conv2->in_channels * model->conv2->kernel_size * model->conv2->kernel_size; i++)
//         model->conv2->weights[i] = 0.01f;

//     for(int i = 0; i < model->conv2->out_channels; i++)
//         model->conv2->bias[i] = 0.0f;

//     for(int i = 0; i < model->bn2->channels; i++)
//     {
//         model->bn2->weight[i] = 1.0f;
//         model->bn2->bias[i] = 0.0f;
//         model->bn2->running_mean[i] = 0.0f;
//         model->bn2->running_variance[i] = 1.0f;
//     }

//     for(int i = 0; i < up->out_channels * up->in_channels * 2 * 2; i++)
//         up->weights[i] = 0.25f;

//     for(int i = 0; i < up->out_channels; i++)
//         up->bias[i] = 0.0f;


//     // Aplicamos la doble convolución
//     Tensor* x1 = DoubleConv2D_Forward(model, input);

//     printf("Tensor x1: %d x %d x %d\n",x1->channels, x1->height, x1->width);

//     // Aplicamos la max pooling. Debe reducir la capa a la mitad.  
//     Tensor* x2 = MaxPool2D_Forward(x1);

//     printf("Tensor x2: %d x %d x %d\n",x2->channels, x2->height, x2->width);

//     // Aplicamos el upsampling. Debe aumentar la capa a la doble.

//     Tensor* x3 = ConvTranspose2D_Forward(up, x2);

//     printf("Tensor x3: %d x %d x %d\n",x3->channels, x3->height, x3->width);
    


//     // Imprimimos el resultado
//     Tensor_Print(x3);

//     // Liberamos memoria
//     Tensor_Destroy(input);
//     Tensor_Destroy(x1);
//     Tensor_Destroy(x2);
//     Tensor_Destroy(x3);

//     DoubleConv2D_Destroy(model);
//     ConvTranspose2D_Destroy(up);

//     return 0;
// }