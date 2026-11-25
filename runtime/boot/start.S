.section .text.init
.global _start


_start:
    csrci mstatus, 8
    csrw mie, zero
    # Get hart ID
    csrr t0, mhartid

    # Enable FPU and accelerator state on each hart
    li t1, 0xa000
    csrs mstatus, t1

    # A trap on any hart is a failure the run reports at once: the reporting
    # hart cannot see that another hart stopped.
    la t1, .Ltrap
    csrw mtvec, t1

    # Set up stack pointer for each hart
    # Each hart gets 4KB stack space (0x1000 bytes)
    li sp, DEVICE_STACK_TOP
    li t1, 0x1000
    mul t1, t0, t1
    sub sp, sp, t1
    # Each hart owns a TLS copy at the top of its 4KB stack slot.
    la t2, __tls_start
    la t3, __tdata_end
    la t4, __tls_end
    sub t5, t4, t2
    addi t5, t5, 15
    andi t5, t5, -16
    sub sp, sp, t5
    mv tp, sp
    mv t6, tp
.Lcopy_tls:
    bgeu t2, t3, .Lzero_tls
    lbu a0, 0(t2)
    sb a0, 0(t6)
    addi t2, t2, 1
    addi t6, t6, 1
    j .Lcopy_tls
.Lzero_tls:
    bgeu t2, t4, .Lrun_main
    andi a0, t6, 7
    beqz a0, .Lzero_tls_words
    sb zero, 0(t6)
    addi t2, t2, 1
    addi t6, t6, 1
    j .Lzero_tls
.Lzero_tls_words:
    addi a0, t2, 8
    bgtu a0, t4, .Lzero_tls_tail
    sd zero, 0(t6)
    addi t2, t2, 8
    addi t6, t6, 8
    j .Lzero_tls_words
.Lzero_tls_tail:
    bgeu t2, t4, .Lrun_main
    sb zero, 0(t6)
    addi t2, t2, 1
    addi t6, t6, 1
    j .Lzero_tls_tail
.Lrun_main:
    # Loaders write only the file image, so .bss can hold a previous run's values on P2E DDR.
    # The reporting hart clears it, then releases the others through the last flag write.
    csrr t0, mhartid
    la t1, .Lbss_ready
    li t2, WORKLOAD_REPORT_HART
    bne t0, t2, .Lwait_bss
    la t2, __bss_start
    la t3, __bss_end
    addi t4, t3, -63
.Lclear_bss_blocks:
    bgeu t2, t4, .Lclear_bss
    sd zero, 0(t2)
    sd zero, 8(t2)
    sd zero, 16(t2)
    sd zero, 24(t2)
    sd zero, 32(t2)
    sd zero, 40(t2)
    sd zero, 48(t2)
    sd zero, 56(t2)
    addi t2, t2, 64
    j .Lclear_bss_blocks
.Lclear_bss:
    bgeu t2, t3, .Lbss_cleared
    sd zero, 0(t2)
    addi t2, t2, 8
    j .Lclear_bss
.Lbss_cleared:
    li t2, 1
    fence rw, w
    sw t2, 0(t1)
    j .Lcall_main
.Lwait_bss:
    lw t2, 0(t1)
    beqz t2, .Lwait_bss
    fence r, rw
.Lcall_main:
    # Jump to main for all harts; the workload dispatches by hart id.
    li a0, 0
    li a1, 0
    call main

    # The explicitly selected reporting hart publishes the result.
    csrr t0, mhartid
    li t1, WORKLOAD_REPORT_HART
    bne t0, t1, 1f
    li t1, 0x60000000
    slli t0, t0, 18
    add t1, t1, t0
    sw a0, 0(t1)
1:
    wfi
    j 1b

# Prints trap context without using the stack, reports 1, and parks the hart.
.balign 4
.Ltrap:
    csrr a0, mhartid
    csrr a1, mcause
    csrr a2, mepc
    csrr a3, mtval
    li t1, 0x60020000
    slli t2, a0, 18
    add t1, t1, t2
    la t2, .Ltrap_message
.Ltrap_text:
    lbu t3, 0(t2)
    beqz t3, .Ltrap_values
    sw t3, 0(t1)
    addi t2, t2, 1
    j .Ltrap_text
.Ltrap_values:
    li t4, 4
.Ltrap_value:
    mv t5, a0
    mv a0, a1
    mv a1, a2
    mv a2, a3
    li t6, 60
.Ltrap_hex:
    srl t2, t5, t6
    andi t2, t2, 15
    li t3, 10
    bltu t2, t3, .Ltrap_digit
    addi t2, t2, 87
    j .Ltrap_emit
.Ltrap_digit:
    addi t2, t2, 48
.Ltrap_emit:
    sw t2, 0(t1)
    addi t6, t6, -4
    bgez t6, .Ltrap_hex
    li t2, 32
    sw t2, 0(t1)
    addi t4, t4, -1
    bnez t4, .Ltrap_value
    li t2, 10
    sw t2, 0(t1)
    csrr t0, mhartid
    li t1, 0x60000000
    slli t0, t0, 18
    add t1, t1, t0
    li t0, 1
    sw t0, 0(t1)
2:
    wfi
    j 2b

.section .rodata
.Ltrap_message:
    .asciz "TRAP hart mcause mepc mtval: "

# Outside .bss so the clear cannot reset it; a stale 1 from a previous run is
# overwritten by the image load because .data is part of the file image.
.section .data
.balign 4
.Lbss_ready:
    .word 0
