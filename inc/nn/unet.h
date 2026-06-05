/**
 * @file unet.h
 * @brief Definición de la arquitectura U-Net.
 *
 * Este archivo contiene la estructura y funciones necesarias
 * para construir una red U-Net basada en bloques de doble
 * convolución, operaciones de pooling y convoluciones transpuestas.
 */

#ifndef UNET_H
#define UNET_H

#include <stdlib.h>

#include "doubleconv2d.h"
#include "maxpool2d.h"
#include "convtranspose2d.h"
#include "conv2d.h"

/**
 * @struct UNet
 * @brief Implementación de una red U-Net.
 *
 * La arquitectura está compuesta por una ruta de codificación
 * (encoder), un cuello de botella (bottleneck) y una ruta de
 * decodificación (decoder) con conexiones de salto (skip connections).
 */
typedef struct
{
    /**
     * @brief Primer bloque del encoder.
     */
    DoubleConv2D* enc1;

    /**
     * @brief Segundo bloque del encoder.
     */
    DoubleConv2D* enc2;

    /**
     * @brief Tercer bloque del encoder.
     */
    DoubleConv2D* enc3;

    /**
     * @brief Cuarto bloque del encoder.
     */
    DoubleConv2D* enc4;

    /**
     * @brief Bloque central de la red.
     */
    DoubleConv2D* bottleneck;

    /**
     * @brief Convolución transpuesta del nivel 4.
     */
    ConvTranspose2D* upconv4;

    /**
     * @brief Bloque decodificador del nivel 4.
     */
    DoubleConv2D* dec4;

    /**
     * @brief Convolución transpuesta del nivel 3.
     */
    ConvTranspose2D* upconv3;

    /**
     * @brief Bloque decodificador del nivel 3.
     */
    DoubleConv2D* dec3;

    /**
     * @brief Convolución transpuesta del nivel 2.
     */
    ConvTranspose2D* upconv2;

    /**
     * @brief Bloque decodificador del nivel 2.
     */
    DoubleConv2D* dec2;

    /**
     * @brief Convolución transpuesta del nivel 1.
     */
    ConvTranspose2D* upconv1;

    /**
     * @brief Bloque decodificador del nivel 1.
     */
    DoubleConv2D* dec1;

    /**
     * @brief Capa convolucional de salida.
     */
    Conv2D* out_conv;

} UNet;

/**
 * @brief Crea e inicializa una red U-Net.
 *
 * Construye todos los bloques del encoder, bottleneck,
 * decoder y la capa de salida.
 *
 * @param in_channels Número de canales de entrada.
 * @param out_channels Número de canales de salida.
 *
 * @return Puntero a la estructura UNet creada.
 * @retval NULL Si ocurre un error de memoria.
 */
UNet* UNet_Create(int in_channels, int out_channels);

/**
 * @brief Libera la memoria asociada a una red U-Net.
 *
 * Libera todos los bloques internos y parámetros de la red.
 *
 * @param this Puntero a la estructura UNet.
 */
void UNet_Destroy(UNet* this);

/**
 * @brief Ejecuta la propagación hacia adelante de la red U-Net.
 *
 * Procesa el tensor de entrada a través del encoder,
 * bottleneck y decoder, generando un tensor de salida.
 *
 * @param this Puntero a la estructura UNet.
 * @param input Tensor de entrada.
 *
 * @return Tensor de salida de la red.
 * @retval NULL Si ocurre un error durante la operación.
 */
Tensor* UNet_Forward(UNet* this, Tensor* input);

#endif