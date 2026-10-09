from examples.balls.mxmm.compiler.python.layout import bank_bytes


def window_bytes(compiler_build):
    return bank_bytes(compiler_build, "ffn")
