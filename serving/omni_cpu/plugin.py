def activate():
    from vllm.platforms import current_platform

    if current_platform.is_cpu():
        return "stack.serving.omni_cpu.platform.Platform"
    return None
