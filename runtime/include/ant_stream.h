#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void ant_emit(uint32_t funct7, uint64_t rs1, uint64_t rs2);
void ant_flush(void);
#ifdef __cplusplus
}
#endif
