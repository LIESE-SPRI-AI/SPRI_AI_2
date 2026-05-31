/**
 * @file tensor.h
 * @brief Definición de la estructura Tensor y operaciones básicas.
 * @author Guillermo Ocaña
 * Este archivo contiene la definición de tensores multidimensionales
 * utilizados en operaciones de redes neuronales y procesamiento numérico.
 */

#ifndef TENSOR_H
#define TENSOR_H

/**
 * @brief Habilita funciones de depuración.
 */
#define DEBUG_MODE

/**
 * @brief Tipo de dato utilizado por el tensor.
 */
typedef float type;

#include <stdlib.h>

#ifdef DEBUG_MODE
#include <stdio.h>
#endif 

/**
 * @struct Tensor
 * @brief Representa un tensor tridimensional.
 *
 * La estructura almacena los datos en formato:
 * channels × height × width.
 */
typedef struct
{
    /**
     * @brief Número de canales del tensor.
     */
    int channels;

    /**
     * @brief Altura del tensor.
     */
    int height;

    /**
     * @brief Anchura del tensor.
     */
    int width;

    /**
     * @brief Número total de elementos almacenados.
     */
    int size;

    /**
     * @brief Puntero a los datos del tensor.
     */
    type * data;

} Tensor;

/**
 * @brief Crea e inicializa un tensor.
 *
 * Reserva memoria para la estructura y sus datos internos.
 *
 * @param c Número de canales.
 * @param h Altura del tensor.
 * @param w Anchura del tensor.
 *
 * @return Puntero al tensor creado.
 * @retval NULL Si ocurre un error de memoria.
 */
Tensor * Tensor_Create(int c, int h, int w);

/**
 * @brief Libera la memoria asociada a un tensor.
 *
 * @param this Puntero al tensor.
 */
void Tensor_Destroy(Tensor * this);

/**
 * @brief Obtiene un elemento del tensor.
 *
 * @param this Puntero al tensor.
 * @param c Índice del canal.
 * @param h Índice de altura.
 * @param w Índice de anchura.
 *
 * @return Valor almacenado en la posición especificada.
 */
type Tensor_Get(Tensor * this, int c, int h, int w);

/**
 * @brief Modifica un elemento del tensor.
 *
 * @param this Puntero al tensor.
 * @param c Índice del canal.
 * @param h Índice de altura.
 * @param w Índice de anchura.
 * @param value Valor a almacenar.
 */
void Tensor_Set(Tensor * this, int c, int h, int w, type value);

/**
 * @brief Concatena dos tensores sobre el eje de canales.
 *
 * Ambos tensores deben tener la misma altura y anchura.
 *
 * @param a Primer tensor.
 * @param b Segundo tensor.
 *
 * @return Nuevo tensor concatenado.
 * @retval NULL Si las dimensiones son incompatibles.
 */
Tensor* Tensor_Concat(Tensor* a, Tensor* b);

    #ifdef DEBUG_MODE

    /**
     * @brief Imprime el contenido del tensor.
     *
     * Función disponible únicamente en modo depuración.
     *
     * @param this Puntero al tensor.
     */
    void Tensor_Print(Tensor * this);

    #endif

void Tensor_LoadImage(Tensor* this, char* path);



#endif