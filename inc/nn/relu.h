/**
 * @file relu.h
 * @brief Implementación de la función de activación ReLU.
 *
 * Este archivo contiene la declaración de la operación
 * Rectified Linear Unit (ReLU) aplicada sobre tensores.
 */

#ifndef RELU_H
#define RELU_H

#include "tensor.h"

/**
 * @brief Aplica la función de activación ReLU sobre un tensor.
 *
 * La operación se realiza en el mismo tensor de entrada.
 * Todos los valores negativos son reemplazados por cero.
 *
 * @param x Tensor de entrada y salida.
 */
void ReLU_Forward(Tensor* x);

#endif