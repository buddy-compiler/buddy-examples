package platform

import chisel3._
import java.io.File

import org.chipsalliance.cde.config.{Config, Parameters, Field}
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.subsystem._
import freechips.rocketchip.devices.tilelink.{BootROMParams, BootROMLocated}
import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.subsystem.SystemBusKey

class WithBootROM extends Config((site, here, up) => {
  case BootROMLocated(x) => {
    val chipyardBootROM = new File(s"../thirdparty/chipyard/generators/testchipip/bootrom/bootrom.rv${site(MaxXLen)}.img")
    val firesimBootROM = new File(s"../thirdparty/chipyard/target-rtl/chipyard/generators/testchipip/bootrom/bootrom.rv${site(MaxXLen)}.img")

    val bootROMPath = if (chipyardBootROM.exists()) {
      chipyardBootROM.getAbsolutePath()
    } else {
      firesimBootROM.getAbsolutePath()
    }
    up(BootROMLocated(x)).map(_.copy(contentFileName = bootROMPath))
  }
})


class GemminiConfig extends Config(
  new gemmini.DefaultGemminiConfig ++
  new freechips.rocketchip.rocket.WithNBigCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

class FireSimGemminiVCU118Config extends Config(
  new WithBootROM ++
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new freechips.rocketchip.subsystem.WithExtMemSize((1 << 30) * 4L) ++
  new GemminiConfig)
  
class FireSimGemminiU280Config extends Config(
  new WithBootROM ++
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new GemminiConfig)
