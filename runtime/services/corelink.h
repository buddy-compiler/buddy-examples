#ifndef BUCKYBALL_CORELINK_H
#define BUCKYBALL_CORELINK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum corelink_field {
  CORELINK_SHARED_BYTES,
  CORELINK_BANK_BYTES,
  CORELINK_CORE_ID,
  CORELINK_CORE_COUNT
};

uint64_t corelink_read64(uint64_t byte_address);
void corelink_write64(uint64_t byte_address, uint64_t data);

uint64_t corelink_query(enum corelink_field field);
void corelink_transfer(uint64_t source_byte_address, uint32_t target_core,
                       uint64_t target_byte_address, uint32_t bytes);

/* Endpoint IDs are zero-based within a core. Export pins an existing group
 * mapping until the consumer finishes and release removes that pin. */
uint64_t corelink_shared_export(uint32_t endpoint, uint32_t virtual_bank,
                                uint32_t group);
void corelink_shared_release(uint32_t endpoint, uint32_t virtual_bank,
                             uint32_t group);

#ifdef __cplusplus
}
#endif

#endif
