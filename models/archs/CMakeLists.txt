set(ARCH_GEMMINI_DIR ${MODELTEST_WORKLOAD_DIR}/archs/gemmini)
set(GEMMINI_DIR ${MODELTEST_WORKLOAD_DIR}/archs/gemmini)
set(ARCH_BUCKYBALL_DIR ${MODELTEST_WORKLOAD_DIR}/archs/buckyball)
set(BUCKYBALL_DIR ${MODELTEST_WORKLOAD_DIR}/archs/buckyball)
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

