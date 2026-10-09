#ifndef BUCKYBALL_TLINK_H
#define BUCKYBALL_TLINK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum tlink_field {
  TLINK_SHARED_BYTES,
  TLINK_BANK_BYTES,
  TLINK_TILE_ID,
  TLINK_TILE_COUNT
};

uint64_t tlink_read64(uint64_t byte_address);
void tlink_write64(uint64_t byte_address, uint64_t data);

uint64_t tlink_query(enum tlink_field field);
void tlink_transfer(uint64_t source_byte_address, uint32_t target_tile,
                    uint64_t target_byte_address, uint32_t bytes);

/* Endpoint IDs are zero-based within a tile. Export pins an existing group
 * mapping until the consumer finishes and release removes that pin. */
uint64_t tlink_shared_export(uint32_t endpoint, uint32_t virtual_bank,
                            uint32_t group);
void tlink_shared_release(uint32_t endpoint, uint32_t virtual_bank,
                          uint32_t group);

#ifdef __cplusplus
}
#endif

#endif
