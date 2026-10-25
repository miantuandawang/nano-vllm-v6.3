"""V5.1 eager input packing and greedy projection; optional KV-only append."""
import torch
import torch.distributed as dist
from nanovllm.utils.context import set_context, reset_context

class V51EagerMethods:
    @torch.inference_mode()
    def _predict(self, seqs, token_lists, all_positions=False, prediction_starts=None, project=True):
        """Append all uncached tokens; project only each query's requested suffix.

        prediction_starts contains query-relative offsets, one per sequence.
        Without it, all_positions selects either the full query or its last row.
        """
        ids, positions, slots = [], [], []
        cu_q, cu_k = [0], [0]
        lengths = []
        for seq, tokens in zip(seqs, token_lists):
            start, end = seq.num_cached_tokens, len(tokens)
            ids.extend(tokens[start:end])
            positions.extend(range(start, end))
            lengths.append(end - start)
            cu_q.append(cu_q[-1] + end - start)
            cu_k.append(cu_k[-1] + end)
            slots.extend(seq.block_table[p // self.block_size] * self.block_size + p % self.block_size
                         if seq.block_table else -1 for p in range(start, end))
        def tensor(values, dtype=torch.int32):
            return torch.tensor(values, dtype=dtype, device="cuda")
        tables = None
        if any(seq.num_cached_tokens for seq in seqs):
            width = max(len(seq.block_table) for seq in seqs)
            tables = tensor([seq.block_table + [-1] * (width - len(seq.block_table)) for seq in seqs])
        set_context(True, tensor(cu_q), tensor(cu_k), max(lengths),
                    max(map(len, token_lists)), tensor(slots), None, tables)
        try:
            hidden = self.model(tensor(ids, torch.int64), tensor(positions, torch.int64))
            if not project:
                return [[] for _ in seqs]
            if prediction_starts is None:
                prediction_starts = [0 if all_positions else length - 1 for length in lengths]
            prediction_counts = [length - start for length, start in zip(lengths, prediction_starts)]
            if any(prediction_starts):
                rows = [row for i, start in enumerate(prediction_starts)
                        for row in range(cu_q[i] + start, cu_q[i + 1])]
                hidden = hidden[tensor(rows, torch.int64)]
            predictions = self._greedy_predictions(hidden)
            results = []
            offset = 0
            for count in prediction_counts:
                results.append(predictions[offset:offset + count])
                offset += count
            return results
        finally:
            reset_context()


    @torch.inference_mode()
    def _greedy_predictions(self, hidden):
        """Vocabulary projection, TP gather/broadcast and CPU results stay outside capture."""
        predictions = []
        for chunk in hidden.split(128):
            logits = self.model.compute_logits(chunk, all_positions=True)
            tokens = (logits.argmax(dim=-1) if self.rank == 0 else
                      torch.empty(chunk.shape[0], dtype=torch.int64, device="cuda"))
            if self.world_size > 1:
                dist.broadcast(tokens, src=0)
            predictions.extend(tokens.tolist())
        return predictions

