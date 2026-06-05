/**
 * @file batchnorm2d.h
 * @brief Definición de la capa Batch Normalization 2D.
 * @author Guillermo Ocaña
 * Este archivo contiene la estructura y funciones necesarias
 * para aplicar normalización por lotes sobre tensores 2D
 * utilizados en redes neuronales convolucionales.
 */

#ifndef BATCHNORM2D_H
#define BATCHNORM2D_H

#include <stdlib.h>
#include <math.h>

#include "tensor.h"

/**
 * @struct BatchNorm2D
 * @brief Representa una capa de Batch Normalization 2D.
 *
 * La capa almacena parámetros aprendibles y estadísticas
 * utilizadas para normalizar activaciones por canal.
 */
typedef struct
{
    /**
     * @brief Número de canales del tensor.
     */
    int channels;

    /**
     * @brief Parámetros de escala por canal.
     */
    type* weight;

    /**
     * @brief Parámetros de desplazamiento por canal.
     */
    type* bias;

    /**
     * @brief Media acumulada por canal.
     */
    type* running_mean;

    /**
     * @brief Varianza acumulada por canal.
     */
    type* running_variance;

    /**
     * @brief Valor pequeño utilizado para estabilidad numérica.
     */
    type epsilon;

} BatchNorm2D;

/**
 * @brief Crea e inicializa una capa BatchNorm2D.
 *
 * Reserva memoria para parámetros y estadísticas internas.
 *
 * @param channels Número de canales de entrada.
 *
 * @return Puntero a la estructura BatchNorm2D creada.
 * @retval NULL Si ocurre un error de memoria.
 */
BatchNorm2D* BatchNorm2D_Create(int channels);

/**
 * @brief Libera la memoria asociada a una capa BatchNorm2D.
 *
 * @param this Puntero a la capa BatchNorm2D.
 */
void BatchNorm2D_Destroy(BatchNorm2D* this);

/**
 * @brief Aplica Batch Normalization sobre un tensor.
 *
 * La normalización se realiza por canal utilizando
 * las estadísticas almacenadas en la capa.
 *
 * @param this Puntero a la capa BatchNorm2D.
 * @param x Tensor de entrada y salida.
 */
void BatchNorm2D_Forward(BatchNorm2D* this, Tensor* x);

#endif