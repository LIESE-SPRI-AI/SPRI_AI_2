#include "nn/relu.h"

void ReLU_Forward(Tensor* x)
{
    type* ptr = x->data;
    type* end = ptr + x->size;

    while(ptr < end)
    {
        if(*ptr < 0.0f)
        {
            *ptr = 0.0f;
        }

        ptr++;
    }
}