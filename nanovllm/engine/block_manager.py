"""V5.1 exclusive-page policy with atomic multi-pool reservation."""
from collections import deque


class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_block_ids = deque(range(num_blocks))

    def reserve(self, seq, required_tokens):
        required = (required_tokens + self.block_size - 1) // self.block_size
        additional = max(0, required - len(seq.block_table))
        if additional > len(self.free_block_ids):
            return False
        seq.block_table.extend(self.free_block_ids.popleft() for _ in range(additional))
        return True

    def truncate(self, seq, valid_cached_tokens):
        seq.num_cached_tokens = min(seq.num_cached_tokens, valid_cached_tokens)
        keep = (seq.num_cached_tokens + self.block_size - 1) // self.block_size
        self.release_tail(seq, keep)

    def release_tail(self, seq, keep):
        self.free_block_ids.extend(seq.block_table[keep:])
        del seq.block_table[keep:]

    def deallocate(self, seq):
        self.free_block_ids.extend(seq.block_table)
        seq.block_table.clear()
        seq.num_cached_tokens = 0


def reserve_atomic(reservations):
    """Reserve a complete batch across both pools or undo newly acquired pages."""
    acquired = []
    for manager, cache, end in reservations:
        before = len(cache.block_table)
        if not manager.reserve(cache, end):
            for previous, state, keep in reversed(acquired):
                previous.release_tail(state, keep)
            return False
        acquired.append((manager, cache, before))
    return True
