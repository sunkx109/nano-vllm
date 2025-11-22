from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i) # 获取i-block的token_ids
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 查询hash_to_block_id字典，找到返回对应block_id，否则返回-1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # 如果cache未命中
                block_id = self.free_block_ids[0] # 取出free的block_ids
                block = self._allocate_block(block_id) # 分配这个block
            else:
                # 如果cache命中了
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 命中了且当前block正在使用中
                    block = self.blocks[block_id] # 直接取
                    block.ref_count += 1 # 引用+1
                else:
                    # 命中，但是没有使用block
                    # 这里其实需要看一下_deallocate_block函数
                    # 释放的时候只是从used_block_ids移除，并没有清空token_ids
                    block = self._allocate_block(block_id)
            if h != -1:
                # 如果计算的hash是有值的，这里要更新一下block与hash_to_block_id的信息
                # 因为有可能虽然缓存命中，但是并不在used_block_ids中
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table # 取出当前seq的block_table
        last_block = self.blocks[block_table[-1]] # 取出当前blockmanager的最后block
        if len(seq) % self.block_size == 1:
            # 需要开启一个新的block时，首先要确保上一个block的hash有值
            assert last_block.hash != -1
            # 取出free block
            block_id = self.free_block_ids[0]
            # allocate 这个block
            self._allocate_block(block_id)
            # 这个seq的block_table要附带上这个block_id
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 这个情况是当前block满了的状态
            # 首先要确保上一个block的hash是有值的
            assert last_block.hash == -1
            # 当前block的所有token_ids
            token_ids = seq.block(seq.num_blocks-1)
            # 如果block_table的长度>1 那就是倒数第二个block的hash
            # 否则就是-1
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix) # 计算当前block的hash
            last_block.update(h, token_ids)# 更新当前block的hash与tokens_id
            self.hash_to_block_id[h] = last_block.block_id # 记录hash与block_id的映射关系
        else:
            # 其余情况下，保持last_block的hash为-1
            assert last_block.hash == -1
