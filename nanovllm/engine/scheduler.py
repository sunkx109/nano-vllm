from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            # 当waiting队列不为空,而且处理num_seqs不超过max_num_seqs
            seq = self.waiting[0] # 取出队列首个seq
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                #如果同时批处理tokens超过最大批处理tokens，或者没有内存分配
                break
            num_seqs += 1
            self.block_manager.allocate(seq) # 给seq分配block
            num_batched_tokens += len(seq) - seq.num_cached_tokens # num_batched_tokens
            seq.status = SequenceStatus.RUNNING # 更改状态
            self.waiting.popleft() #从waiting队列出
            self.running.append(seq) #进running队列
            scheduled_seqs.append(seq) # 同时标记为被调度的seq
        #优先处理waiting队列直到不能处理后返回scheduled_seqs
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        
        while self.running and num_seqs < self.max_num_seqs:
		        # 当running队列不为空，而且处理num_seqs不超过max_num_seqs
            seq = self.running.popleft() #取出running队列左侧的seq
            while not self.block_manager.can_append(seq):
                # 如果不能为seq分配新的block
                if self.running:
                    # running队列不为空，就抢占队列最后的seq
                    self.preempt(self.running.pop())
                else:
                    # 否则就抢占自己
                    self.preempt(seq)
                    break
            else:
                # 如果能分配block
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 保序将scheduled_seqs插入running队列前面
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        # 更改seq的状态为waiting
        # 同时销毁seq的block资源，这意味这个seq即使已经经过prefill，但是现在开始也会被打回重算
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
