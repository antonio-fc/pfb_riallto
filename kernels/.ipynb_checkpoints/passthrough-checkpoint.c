#include <stdint.h>

void passthrough(uint16_t *in_buffer, uint16_t *out_buffer, uint32_t nbytes)
{
    for(int i=0; i<nbytes; i++) {
        out_buffer[i] = in_buffer[i];
    }
}