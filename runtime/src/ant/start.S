.section .text.start,"ax",@progbits
.globl _start
.type _start,@function
_start:
  call ant_main
  li a7, 0
  ecall
  ebreak
.size _start, .-_start
