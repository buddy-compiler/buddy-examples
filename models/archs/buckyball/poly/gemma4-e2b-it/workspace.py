class Pool:
    def __init__(self, capacity):
        self.blocks = [[0, capacity - 48, True]]
        self.live = {}
        self.high_water = 0
        self.used = self.peak = 0

    def allocate(self, name, size):
        size = (size + 15) // 16 * 16
        for index, block in enumerate(self.blocks):
            start, available, free = block
            if not free or available < size:
                continue
            if available - size >= 64:
                self.blocks.insert(
                    index + 1, [start + 48 + size, available - size - 48, True]
                )
                block[1] = size
            block[2] = False
            self.live[name] = block
            self.used += block[1] + 48
            self.peak = max(self.peak, self.used)
            self.high_water = max(self.high_water, start + 48 + block[1])
            return
        raise ValueError("workspace allocation exceeds the planned arena")

    def free(self, name):
        block = self.live.pop(name)
        index = self.blocks.index(block)
        self.used -= block[1] + 48
        block[2] = True
        if index + 1 < len(self.blocks) and self.blocks[index + 1][2]:
            following = self.blocks.pop(index + 1)
            block[1] += 48 + following[1]
        if index and self.blocks[index - 1][2]:
            self.blocks[index - 1][1] += 48 + block[1]
            self.blocks.pop(index)
