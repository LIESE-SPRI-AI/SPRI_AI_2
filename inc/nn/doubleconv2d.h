/**
 * @file doubleconv2d.h
 * @brief Definición del bloque Double Convolution 2D.
 * @author Guillermo Ocaña
 * Este archivo contiene la estructura y funciones necesarias
 * para implementar un bloque compuesto por dos convoluciones
 * consecutivas con Batch Normalization y activación ReLU.
 */

#ifndef DOUBLECONV2D_H
#define DOUBLECONV2D_H

#include <stdlib.h>

#include "nn/conv2d.h"
#include "nn/batchnorm2d.h"
#include "nn/relu.h"

/**
 * @struct DoubleConv2D
 * @brief Representa un bloque de doble convolución.
 *
 * El bloque aplica la siguiente secuencia:
 *
 * Conv2D -> BatchNorm2D -> ReLU ->
 * Conv2D -> BatchNorm2D -> ReLU
 */
typedef struct
{
    /**
     * @brief Primera capa convolucional.
     */
    Conv2D* conv1;

    /**
     * @brief Primera capa Batch Normalization.
     */
    BatchNorm2D* bn1;

    /**
     * @brief Segunda capa convolucional.
     */
    Conv2D* conv2;

    /**
     * @brief Segunda capa Batch Normalization.
     */
    BatchNorm2D* bn2;

} DoubleConv2D;

/**
 * @brief Crea e inicializa un bloque DoubleConv2D.
 *
 * Inicializa dos capas convolucionales y sus respectivas
 * capas Batch Normalization.
 *
 * @param in_channels Número de canales de entrada.
 * @param out_channels Número de canales de salida.
 *
 * @return Puntero al bloque DoubleConv2D creado.
 * @retval NULL Si ocurre un error de memoria.
 */
DoubleConv2D* DoubleConv2D_Create(int in_channels, int out_channels);

/**
 * @brief Libera la memoria asociada a un bloque DoubleConv2D.
 *
 * @param this Puntero al bloque DoubleConv2D.
 */
void DoubleConv2D_Destroy(DoubleConv2D* this);

/**
 * @brief Ejecuta la propagación hacia adelante del bloque.
 *
 * Aplica dos convoluciones consecutivas con Batch
 * Normalization y activación ReLU.
 *
 * @param this Puntero al bloque DoubleConv2D.
 * @param input Tensor de entrada.
 *
 * @return Tensor de salida procesado.
 * @retval NULL Si ocurre un error durante la operación.
 */
Tensor* DoubleConv2D_Forward(DoubleConv2D* this, Tensor* input);

#endif