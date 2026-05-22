/**
 * @file conv2d.h
 * @brief Definición de la capa de convolución 2D.
 *
 * Este archivo contiene la estructura y las funciones necesarias
 * para crear, utilizar y destruir una capa convolucional 2D
 * utilizada en redes neuronales.
 */

#ifndef CONV2D_H
#define CONV2D_H

#include "tensor.h"

/**
 * @struct Conv2D
 * @brief Representa una capa de convolución 2D.
 *
 * La estructura almacena los parámetros de configuración de la capa,
 * así como los pesos y sesgos asociados.
 */
typedef struct
{
    /**
     * @brief Número de canales de entrada.
     */
    int in_channels;

    /**
     * @brief Número de canales de salida.
     */
    int out_channels;

    /**
     * @brief Tamaño del kernel convolucional.
     *
     * Se asume kernel cuadrado de tamaño
     * kernel_size x kernel_size.
     */
    int kernel_size;

    /**
     * @brief Paso de desplazamiento de la convolución.
     */
    int stride;

    /**
     * @brief Cantidad de padding aplicado a la entrada.
     */
    int padding;

    /**
     * @brief Arreglo de pesos de la convolución.
     */
    type* weights;

    /**
     * @brief Arreglo de sesgos de la convolución.
     */
    type* bias;

} Conv2D;

/**
 * @brief Crea e inicializa una capa Conv2D.
 *
 * Reserva memoria para la estructura y sus parámetros internos.
 *
 * @param in_channels Número de canales de entrada.
 * @param out_channels Número de canales de salida.
 * @param kernel_size Tamaño del kernel.
 * @param stride Paso de convolución.
 * @param padding Padding aplicado a la entrada.
 *
 * @return Puntero a la estructura Conv2D creada.
 * @retval NULL Si ocurre un error de memoria.
 */
Conv2D* Conv2D_Create(int in_channels,int out_channels,int kernel_size,int stride,int padding);

/**
 * @brief Libera la memoria asociada a una capa Conv2D.
 *
 * También libera los pesos y sesgos almacenados.
 *
 * @param this Puntero a la capa Conv2D.
 */
void Conv2D_Destroy(Conv2D* this);

/**
 * @brief Ejecuta la propagación de la convolución.
 *
 * Aplica la operación convolucional sobre el tensor de entrada
 * utilizando los pesos y sesgos de la capa.
 *
 * @param this Puntero a la capa Conv2D.
 * @param input Tensor de entrada.
 *
 * @return Tensor de salida generado por la convolución.
 * @retval NULL Si ocurre un error durante la operación.
 */
Tensor* Conv2D_Forward( Conv2D* this, Tensor* input);

#endif