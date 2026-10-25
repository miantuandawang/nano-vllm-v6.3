"""V5.1 graph functions, reused with per-query-length owners in V6.0.
See PROVENANCE.json for source hashes.
"""
import torch
import torch.distributed as dist
from nanovllm.utils.context import set_context, reset_context


class V51GraphMethods:
    def _set_verify_graph_context(self, bucket):
        buffers = self.verify_graph_vars
        num_tokens = bucket * self.verify_query_len
        set_context(
            True,
            cu_seqlens_q=buffers["cu_seqlens_q"][:bucket + 1],
            cu_seqlens_k=buffers["cu_seqlens_k"][:bucket + 1],
            max_seqlen_q=self.verify_query_len,
            max_seqlen_k=self.config.max_model_len,
            slot_mapping=buffers["slot_mapping"][:num_tokens],
            block_tables=buffers["block_tables"][:bucket],
        )


    @torch.inference_mode()
    def capture_verify_graphs(self):
        """Native batch buckets/shared pool, with K+1 query tokens per request."""
        max_bs = self.verify_graph_max_bs
        qlen = self.verify_query_len
        max_tokens = max_bs * qlen
        max_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.verify_graph_bs = sorted({bs for bs in sizes if bs <= max_bs} | {max_bs})
        def zeros(*shape, dtype=torch.int32):
            return torch.zeros(*shape, dtype=dtype, device="cuda")
        # Scratch IDs are outside the allocator's [0, num_kvcache_blocks) range.
        scratch_start = self.config.num_kvcache_blocks
        block_tables = torch.full((max_bs, max_blocks), scratch_start,
                                  dtype=torch.int32, device="cuda")
        for column in range(self.verify_scratch_blocks):
            block_tables[:, column] = scratch_start + column
        self.verify_graph_vars = dict(
            input_ids=zeros(max_tokens, dtype=torch.int64),
            positions=zeros(max_tokens, dtype=torch.int64),
            slot_mapping=torch.full((max_tokens,), -1, dtype=torch.int32, device="cuda"),
            block_tables=block_tables,
            cu_seqlens_q=torch.arange(max_bs + 1, dtype=torch.int32, device="cuda") * qlen,
            cu_seqlens_k=torch.arange(max_bs + 1, dtype=torch.int32, device="cuda") * qlen,
            outputs=zeros(max_tokens, self.config.hf_config.hidden_size,
                          dtype=self.config.hf_config.dtype),
        )
        self.verify_graphs = {}
        self.verify_graph_pool = None
        buffers = self.verify_graph_vars
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        try:
            for bs in reversed(self.verify_graph_bs):
                count = bs * qlen
                self._set_verify_graph_context(bs)
                with torch.cuda.stream(warmup_stream):
                    for _ in range(3):
                        buffers["outputs"][:count] = self.model(
                            buffers["input_ids"][:count], buffers["positions"][:count])
                torch.cuda.current_stream().wait_stream(warmup_stream)
                torch.cuda.synchronize()
                # All TP ranks capture the same collective sequence and shape.
                if self.world_size > 1:
                    dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self.verify_graph_pool):
                    buffers["outputs"][:count] = self.model(
                        buffers["input_ids"][:count], buffers["positions"][:count])
                if self.verify_graph_pool is None:
                    self.verify_graph_pool = graph.pool()
                self.verify_graphs[bs] = graph
                torch.cuda.synchronize()
        finally:
            reset_context()


    def prepare_verify_graph_inputs(self, seqs):
        bs = len(seqs)
        bucket = next(size for size in self.verify_graph_bs if size >= bs)
        qlen = self.verify_query_len
        count = bucket * qlen
        buffers = self.verify_graph_vars
        scratch_start = self.config.num_kvcache_blocks
        # Clear the WHOLE bucket to handle shrinking batches and shorter tables.
        buffers["input_ids"][:count].zero_()
        buffers["positions"][:count].zero_()
        buffers["slot_mapping"][:count].fill_(-1)
        buffers["block_tables"][:bucket].fill_(scratch_start)
        for column in range(self.verify_scratch_blocks):
            buffers["block_tables"][:bucket, column].fill_(scratch_start + column)

        ids, positions, slots = [], [], []
        cu_k = [0]
        width = max(len(seq.block_table) for seq in seqs)
        tables = []
        for seq in seqs:
            tokens = seq.token_ids + seq.draft_tokens
            start, end = seq.num_cached_tokens, len(tokens)
            ids.extend(tokens[start:end])
            positions.extend(range(start, end))
            slots.extend(seq.block_table[p // self.block_size] * self.block_size + p % self.block_size
                         for p in range(start, end))
            cu_k.append(cu_k[-1] + end)
            tables.append(seq.block_table + [scratch_start] * (width - len(seq.block_table)))
        # Dummy rows attend to qlen zero KV entries, never real requests' blocks.
        for _ in range(bucket - bs):
            cu_k.append(cu_k[-1] + qlen)

        def upload(destination, values, dtype=torch.int32):
            cpu = torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True)
            destination.copy_(cpu, non_blocking=True)
        upload(buffers["input_ids"][:bs * qlen], ids, torch.int64)
        upload(buffers["positions"][:bs * qlen], positions, torch.int64)
        upload(buffers["slot_mapping"][:bs * qlen], slots)
        upload(buffers["cu_seqlens_k"][:bucket + 1], cu_k)
        upload(buffers["block_tables"][:bs, :width], tables)
        return bucket


    @torch.inference_mode()
    def replay_verify_graph(self, seqs):
        bucket = self.prepare_verify_graph_inputs(seqs)
        self._set_verify_graph_context(bucket)
        try:
            self.verify_graphs[bucket].replay()
            qlen = self.verify_query_len
            # Padding hidden states never enter the LM head or acceptance logic.
            predicted = self._greedy_predictions(self.verify_graph_vars["outputs"][:len(seqs) * qlen])
            return [predicted[i * qlen:(i + 1) * qlen] for i in range(len(seqs))]
        finally:
            reset_context()


    def _set_catchup_graph_context(self, bucket):
        buffers = self.catchup_graph_vars
        num_tokens = bucket * self.catchup_query_len
        set_context(
            True,
            cu_seqlens_q=buffers["cu_seqlens_q"][:bucket + 1],
            cu_seqlens_k=buffers["cu_seqlens_k"][:bucket + 1],
            max_seqlen_q=self.catchup_query_len,
            max_seqlen_k=self.config.max_model_len,
            slot_mapping=buffers["slot_mapping"][:num_tokens],
            block_tables=buffers["block_tables"][:bucket],
        )


    @torch.inference_mode()
    def capture_catchup_graphs(self):
        """Two uncached tokens per request; separate pool from single-token decode."""
        max_bs = self.catchup_graph_max_bs
        qlen = self.catchup_query_len
        max_tokens = max_bs * qlen
        max_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.catchup_graph_bs = sorted({bs for bs in sizes if bs <= max_bs} | {max_bs})
        def zeros(*shape, dtype=torch.int32):
            return torch.zeros(*shape, dtype=dtype, device="cuda")
        # Scratch IDs are outside the allocator's [0, num_kvcache_blocks) range.
        scratch_start = self.config.num_kvcache_blocks
        block_tables = torch.full((max_bs, max_blocks), scratch_start,
                                  dtype=torch.int32, device="cuda")
        for column in range(self.catchup_scratch_blocks):
            block_tables[:, column] = scratch_start + column
        self.catchup_graph_vars = dict(
            input_ids=zeros(max_tokens, dtype=torch.int64),
            positions=zeros(max_tokens, dtype=torch.int64),
            slot_mapping=torch.full((max_tokens,), -1, dtype=torch.int32, device="cuda"),
            block_tables=block_tables,
            cu_seqlens_q=torch.arange(max_bs + 1, dtype=torch.int32, device="cuda") * qlen,
            cu_seqlens_k=torch.arange(max_bs + 1, dtype=torch.int32, device="cuda") * qlen,
            outputs=zeros(max_tokens, self.config.hf_config.hidden_size,
                          dtype=self.config.hf_config.dtype),
        )
        self.catchup_graphs = {}
        self.catchup_graph_pool = None
        buffers = self.catchup_graph_vars
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        try:
            for bs in reversed(self.catchup_graph_bs):
                count = bs * qlen
                self._set_catchup_graph_context(bs)
                with torch.cuda.stream(warmup_stream):
                    for _ in range(3):
                        buffers["outputs"][:count] = self.model(
                            buffers["input_ids"][:count], buffers["positions"][:count])
                torch.cuda.current_stream().wait_stream(warmup_stream)
                torch.cuda.synchronize()
                # All TP ranks capture the same collective sequence and shape.
                if self.world_size > 1:
                    dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self.catchup_graph_pool):
                    buffers["outputs"][:count] = self.model(
                        buffers["input_ids"][:count], buffers["positions"][:count])
                if self.catchup_graph_pool is None:
                    self.catchup_graph_pool = graph.pool()
                self.catchup_graphs[bs] = graph
                torch.cuda.synchronize()
        finally:
            reset_context()


    def prepare_catchup_graph_inputs(self, seqs, token_lists):
        if not seqs or len(seqs) != len(token_lists):
            raise ValueError("Catch-up graph requires matching nonempty inputs")
        if any(len(tokens) - seq.num_cached_tokens != 2
               for seq, tokens in zip(seqs, token_lists)):
            raise ValueError("Catch-up graph requires exactly two uncached tokens per row")
        bs = len(seqs)
        bucket = next(size for size in self.catchup_graph_bs if size >= bs)
        qlen = self.catchup_query_len
        count = bucket * qlen
        buffers = self.catchup_graph_vars
        scratch_start = self.config.num_kvcache_blocks
        # Clear the WHOLE bucket to handle shrinking batches and shorter tables.
        buffers["input_ids"][:count].zero_()
        buffers["positions"][:count].zero_()
        buffers["slot_mapping"][:count].fill_(-1)
        buffers["block_tables"][:bucket].fill_(scratch_start)
        for column in range(self.catchup_scratch_blocks):
            buffers["block_tables"][:bucket, column].fill_(scratch_start + column)

        ids, positions, slots = [], [], []
        cu_k = [0]
        width = max(len(seq.block_table) for seq in seqs)
        tables = []
        for seq, tokens in zip(seqs, token_lists):
            start, end = seq.num_cached_tokens, len(tokens)
            ids.extend(tokens[start:end])
            positions.extend(range(start, end))
            slots.extend(seq.block_table[p // self.block_size] * self.block_size + p % self.block_size
                         for p in range(start, end))
            cu_k.append(cu_k[-1] + end)
            tables.append(seq.block_table + [scratch_start] * (width - len(seq.block_table)))
        # Dummy rows attend to qlen zero KV entries, never real requests' blocks.
        for _ in range(bucket - bs):
            cu_k.append(cu_k[-1] + qlen)

        def upload(destination, values, dtype=torch.int32):
            cpu = torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True)
            destination.copy_(cpu, non_blocking=True)
        upload(buffers["input_ids"][:bs * qlen], ids, torch.int64)
        upload(buffers["positions"][:bs * qlen], positions, torch.int64)
        upload(buffers["slot_mapping"][:bs * qlen], slots)
        upload(buffers["cu_seqlens_k"][:bucket + 1], cu_k)
        upload(buffers["block_tables"][:bs, :width], tables)
        return bucket


    @torch.inference_mode()
    def replay_catchup_graph(self, seqs, token_lists):
        bucket = self.prepare_catchup_graph_inputs(seqs, token_lists)
        self._set_catchup_graph_context(bucket)
        try:
            self.catchup_graphs[bucket].replay()
            qlen = self.catchup_query_len
            # Both query rows write KV; only the final row predicts the next draft token.
            hidden = self.catchup_graph_vars["outputs"][qlen - 1:len(seqs) * qlen:qlen]
            return [[token] for token in self._greedy_predictions(hidden)]
        finally:
            reset_context()


    @torch.inference_mode()
    def capture_cudagraph(self):
        """Native-style single-token backbone capture, largest bucket first."""
        max_bs = min(self.config.max_num_seqs, 512)
        max_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        buckets = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graph_bs = sorted({bs for bs in buckets if bs <= max_bs} | {max_bs})
        def zeros(*shape, dtype=torch.int32):
            return torch.zeros(*shape, dtype=dtype, device="cuda")
        self.graph_vars = dict(
            input_ids=zeros(max_bs, dtype=torch.int64),
            positions=zeros(max_bs, dtype=torch.int64),
            slot_mapping=torch.full((max_bs,), -1, dtype=torch.int32, device="cuda"),
            context_lens=torch.ones(max_bs, dtype=torch.int32, device="cuda"),
            block_tables=torch.full((max_bs, max_blocks), self.config.num_kvcache_blocks,
                                    dtype=torch.int32, device="cuda"),
            outputs=zeros(max_bs, self.config.hf_config.hidden_size,
                          dtype=self.config.hf_config.dtype),
        )
        self.graphs = {}
        self.graph_pool = None
        buffers = self.graph_vars
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        try:
            for bs in reversed(self.graph_bs):
                set_context(False, slot_mapping=buffers["slot_mapping"][:bs],
                            context_lens=buffers["context_lens"][:bs],
                            block_tables=buffers["block_tables"][:bs])
                # Compile/initialize kernels outside capture, on a side stream.
                with torch.cuda.stream(warmup_stream):
                    for _ in range(3):
                        buffers["outputs"][:bs] = self.model(
                            buffers["input_ids"][:bs], buffers["positions"][:bs])
                torch.cuda.current_stream().wait_stream(warmup_stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self.graph_pool):
                    buffers["outputs"][:bs] = self.model(
                        buffers["input_ids"][:bs], buffers["positions"][:bs])
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[bs] = graph
                torch.cuda.synchronize()
        finally:
            reset_context()


    def prepare_decode(self, seqs, token_lists):
        """Fill static buffers; the caller guarantees one uncached token per row."""
        bs = len(seqs)
        bucket = next(size for size in self.graph_bs if size >= bs)
        buffers = self.graph_vars
        positions = [len(tokens) - 1 for tokens in token_lists]
        slots = [seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size
                 for seq, pos in zip(seqs, positions)]
        width = max(len(seq.block_table) for seq in seqs)
        scratch = self.config.num_kvcache_blocks
        tables = [seq.block_table + [scratch] * (width - len(seq.block_table)) for seq in seqs]

        # Reset the entire replay bucket so shrinking batches cannot reuse old rows.
        buffers["input_ids"][:bucket].zero_()
        buffers["positions"][:bucket].zero_()
        buffers["slot_mapping"][:bucket].fill_(-1)
        buffers["context_lens"][:bucket].fill_(1)
        buffers["block_tables"][:bucket].fill_(scratch)
        def copy(name, values, dtype=torch.int32):
            cpu = torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True)
            if name == "block_tables":
                buffers[name][:bs, :width].copy_(cpu, non_blocking=True)
            else:
                buffers[name][:bs].copy_(cpu, non_blocking=True)
        copy("input_ids", [tokens[-1] for tokens in token_lists], torch.int64)
        copy("positions", positions, torch.int64)
        copy("slot_mapping", slots)
        copy("context_lens", [len(tokens) for tokens in token_lists])
        copy("block_tables", tables)
        return bucket


    @torch.inference_mode()
    def replay_decode_graph(self, seqs, token_lists):
        bs = len(seqs)
        bucket = self.prepare_decode(seqs, token_lists)
        buffers = self.graph_vars
        set_context(False, slot_mapping=buffers["slot_mapping"][:bucket],
                    context_lens=buffers["context_lens"][:bucket],
                    block_tables=buffers["block_tables"][:bucket])
        try:
            self.graphs[bucket].replay()
            # Match upstream capture boundaries: vocabulary projection and sampling
            # stay outside the graph and operate only on real requests.
            logits = self.model.compute_logits(buffers["outputs"][:bs], all_positions=True)
            tokens = (logits.argmax(dim=-1) if self.rank == 0 else
                      torch.empty(bs, dtype=torch.int64, device="cuda"))
            if self.world_size > 1:
                dist.broadcast(tokens, src=0)
            return [[token] for token in tokens.tolist()]
        finally:
            reset_context()

