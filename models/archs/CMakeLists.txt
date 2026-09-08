set(ARCH_GEMMINI_DIR ${ARCH_DIR}/gemmini)
set(GEMMINI_DIR ${ARCH_DIR}/gemmini)
set(ARCH_BUCKYBALL_DIR ${ARCH_DIR}/buckyball)
set(BUCKYBALL_ROOT_DIR ${ARCH_DIR}/buckyball)
set(ARCH_NV_DIR ${ARCH_DIR}/nv)
set(NV_DIR ${ARCH_DIR}/nv)

if (ARCH_GEMMINI)
  add_subdirectory(gemmini)
endif()

if (ARCH_BUCKYBALL)
  add_subdirectory(buckyball)
endif()

if (ARCH_NV)
  add_subdirectory(nv)
endif()
