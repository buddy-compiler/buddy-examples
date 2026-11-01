package platform

import org.chipsalliance.cde.config.Config
import gemmini.CapacityInKilobytes

class GemminiConfig extends Config(
  new gemmini.DefaultGemminiConfig ++
  new freechips.rocketchip.rocket.WithNBigCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class FireSimGemminiVCU118Config extends Config(
  new freechips.rocketchip.subsystem.WithExtMemSize((1 << 30) * 4L) ++
  new chipyard.config.WithBootROM ++
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new GemminiConfig)

class FireSimGemminiU280Config extends Config(
  new chipyard.config.WithBootROM ++
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new GemminiConfig)

//===----------------------------------------------------------===//
// Gemmini Dim 32 Config
//===----------------------------------------------------------===//
class GemminiDim32Config extends Config(
  new gemmini.DefaultGemminiConfig(gemminiConfig = gemmini.GemminiConfigs.defaultConfig.copy(
    meshRows = 32,
    meshColumns = 32,
    has_training_convs = false,
    sp_capacity = CapacityInKilobytes(2048),
    acc_capacity = CapacityInKilobytes(512),
    has_normalizations = true,
  )) ++
  new freechips.rocketchip.rocket.WithNBigCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class FireSimGemminiU280LargeDimConfig extends Config(
  new chipyard.config.WithBootROM ++
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new GemminiDim32Config)
