from stack.runtime.scripts.chip_targets import bank_bytes


def window_bytes(chip_pb):
    return bank_bytes(chip_pb, "ffn")
