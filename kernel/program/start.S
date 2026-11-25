.section .text.start,"ax",@progbits
.globl _start
.type _start,@function
_start:
  call entry
  li a7, 0
  ecall
  ebreak
.size _start, .-_start

# issue(funct7, rs1, rs2): jumps to the 8-byte stub of funct7, which issues
# the Ball instruction and returns. A computed jump: thread code has no data
# to load a jump table from.
.text
.globl issue
.type issue,@function
.balign 8
issue:
  auipc t0, 0
  slli a0, a0, 3
  add t0, t0, a0
  jr 16(t0)
.balign 8
stubs:
.set funct, 0
.rept 128
  .insn r 0x7b, 3, funct, x0, a1, a2
  ret
.set funct, funct + 1
.endr
.size issue, .-issue
