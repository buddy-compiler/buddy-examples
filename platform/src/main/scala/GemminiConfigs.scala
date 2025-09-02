package platform

import chisel3._
import org.chipsalliance.cde.config.{Config, Parameters, Field}
import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.subsystem.SystemBusKey

class GemminiConfig extends Config(
  new gemmini.DefaultGemminiConfig ++
  new freechips.rocketchip.rocket.WithNBigCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)

class FireSimGemminiVCU118Config extends Config(
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new freechips.rocketchip.subsystem.WithExtMemSize((1 << 30) * 4L) ++
  new chipyard.GemminiConfig)
  