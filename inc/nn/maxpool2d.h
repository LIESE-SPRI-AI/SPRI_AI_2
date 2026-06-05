/**
 * @file maxpool2d.h
 * @brief Implementación de la operación Max Pooling 2D.
 * @author Guillermo Ocaña
 * 
 * Este archivo contiene la declaración de la operación
 * Max Pooling utilizada para reducir la resolución espacial
 * de mapas de características en redes neuronales convolucionales.
 */

#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#include "tensor.h"

/**
 * @brief Aplica una operación Max Pooling 2D sobre un tensor.
 *
 * Para cada región de pooling, se selecciona el valor máximo,
 * reduciendo las dimensiones espaciales del tensor de entrada.
 *
 * @param input Tensor de entrada.
 *
 * @return Tensor resultante después de la operación de Max Pooling.
 * @retval NULL Si ocurre un error durante la operación.
 */
Tensor* MaxPool2D_Forward(Tensor* input);

#endif