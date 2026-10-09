OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS {
  . = 0;
  .text : { KEEP(*(.text.start)) *(.text .text.*) . = ALIGN(4); }
  .unsupported : { *(.rodata .rodata.* .srodata .srodata.* .data .data.* .sdata .sdata.* .bss .bss.* .sbss .sbss.* COMMON) }
  ASSERT(SIZEOF(.unsupported) == 0, "Ant code image requires explicit local data initialization")
  /DISCARD/ : { *(.comment .eh_frame .eh_frame.*) }
}
