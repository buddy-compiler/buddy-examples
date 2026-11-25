# The parameters of Qwen3-0.6B in the program of the device: the FP16 weights
# of the Linears below the firmware of the tiles, the FP32 parameters above it
# (the device's device.ld).
.section .parameters,"a",@progbits
.balign 64
.global weight_data
.global weight_data_end
weight_data:
.incbin "arg0.data"
weight_data_end:

.section .parameters.high,"a",@progbits
.balign 64
.global parameter_data
.global parameter_data_end
parameter_data:
.incbin "arg1.data"
parameter_data_end:
