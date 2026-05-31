/**
 * @file convtranspose2d.h
 * @brief Definición de la capa de convolución transpuesta 2D.
 *
 * Este archivo contiene la estructura y funciones necesarias
 * para realizar operaciones de upsampling mediante convolución
 * transpuesta, comúnmente utilizadas en arquitecturas tipo U-Net.
 */

#ifndef CONVTRANSPOSE2D_H
#define CONVTRANSPOSE2D_H

#include <stdlib.h>

#include "tensor.h"

/**
 * @struct ConvTranspose2D
 * @brief Representa una capa de convolución transpuesta 2D.
 *
 * La capa almacena los parámetros necesarios para transformar
 * mapas de características de entrada en mapas de mayor resolución.
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
     * @brief Arreglo de pesos de la convolución transpuesta.
     */
    type* weights;

    /**
     * @brief Arreglo de sesgos de la convolución transpuesta.
     */
    type* bias;

} ConvTranspose2D;

/**
 * @brief Crea e inicializa una capa ConvTranspose2D.
 *
 * Reserva memoria para la estructura y sus parámetros internos.
 *
 * @param in_channels Número de canales de entrada.
 * @param out_channels Número de canales de salida.
 *
 * @return Puntero a la estructura ConvTranspose2D creada.
 * @retval NULL Si ocurre un error de memoria.
 */
ConvTranspose2D* ConvTranspose2D_Create(int in_channels, int out_channels);

/**
 * @brief Libera la memoria asociada a una capa ConvTranspose2D.
 *
 * También libera los pesos y sesgos almacenados.
 *
 * @param this Puntero a la capa ConvTranspose2D.
 */
void ConvTranspose2D_Destroy(ConvTranspose2D* this);

/**
 * @brief Ejecuta la propagación hacia adelante de la convolución transpuesta.
 *
 * Aplica una operación de upsampling mediante convolución
 * transpuesta sobre el tensor de entrada.
 *
 * @param this Puntero a la capa ConvTranspose2D.
 * @param input Tensor de entrada.
 *
 * @return Tensor de salida generado por la convolución transpuesta.
 * @retval NULL Si ocurre un error durante la operación.
 */
Tensor* ConvTranspose2D_Forward(ConvTranspose2D* this, Tensor* input);

#endif