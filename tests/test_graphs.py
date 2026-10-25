"""V5.1 packing regressions reused against the V6.0 graph methods (CPU only)."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
from collections import Counter, defaultdict
ROOT = Path(__file__).resolve().parents[1]

def methods(*names, **globals_):
    tree = ast.parse((ROOT / 'nanovllm/engine/graph_methods.py').read_text(encoding='utf-8'))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    for n in nodes:
        n.decorator_list = []
    ns = dict(globals_)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual ModelRunner>', 'exec'), ns)
    return ns


class Buffer:
    """Small CPU tensor view to check actual packing without a CUDA dependency."""
    def __init__(self, data, rows=None, cols=None, matrix=None):
        self.data = data
        self.rows = list(range(len(data))) if rows is None else rows
        self.matrix = bool(data and isinstance(data[0], list)) if matrix is None else matrix
        self.cols = list(range(len(data[0]))) if cols is None and self.matrix else cols

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            rows = self.rows[row]
            cols = self.cols[col]
            return Buffer(self.data, rows if isinstance(rows, list) else [rows],
                          cols if isinstance(cols, list) else [cols], matrix=True)
        rows = self.rows[key]
        return Buffer(self.data, rows if isinstance(rows, list) else [rows], self.cols, self.matrix)

    def fill_(self, value):
        for row in self.rows:
            if self.matrix:
                for col in self.cols:
                    self.data[row][col] = value
            else:
                self.data[row] = value
        return self

    def zero_(self):
        return self.fill_(0)

    def copy_(self, values, non_blocking=False):
        assert len(values) == len(self.rows)
        for row, value in zip(self.rows, values):
            if self.matrix:
                # A column fill uses fill_ above; upload is always a matrix.
                assert len(value) == len(self.cols)
                for col, cell in zip(self.cols, value):
                    self.data[row][col] = cell
            else:
                self.data[row] = value
        return self

    def tolist(self):
        return [[self.data[r][c] for c in self.cols] for r in self.rows] if self.matrix else [self.data[r] for r in self.rows]


class PackingTests(unittest.TestCase):
    def runner(self):
        vectors={n:Buffer([999]*16) for n in ['input_ids','positions','slot_mapping']}
        vectors['cu_seqlens_k']=Buffer([999]*9)
        vectors['block_tables']=Buffer([[999]*4 for _ in range(8)])
        r=NS(catchup_graph_bs=[1,2,4,8],catchup_query_len=2,catchup_scratch_blocks=1,
             catchup_graph_vars=vectors,config=NS(num_kvcache_blocks=100),block_size=256)
        torch=NS(tensor=lambda values,**kw:values,int32='i32',int64='i64')
        prepare=methods('prepare_catchup_graph_inputs',torch=torch)['prepare_catchup_graph_inputs']
        r.prepare=lambda seqs,tokens:prepare(r,seqs,tokens)
        return r

    def test_page_boundary_positions_slots_and_padded_row_are_correct(self):
        r=self.runner()
        seqs=[NS(num_cached_tokens=255,block_table=[7,9]),
              NS(num_cached_tokens=8,block_table=[11]),NS(num_cached_tokens=1,block_table=[12])]
        tokens=[list(range(257)),list(range(10)),list(range(3))]
        self.assertEqual(r.prepare(seqs,tokens),4)
        b=r.catchup_graph_vars
        self.assertEqual(b['positions'][:8].tolist(),[255,256,8,9,1,2,0,0])
        self.assertEqual(b['slot_mapping'][:8].tolist(),[7*256+255,9*256,11*256+8,11*256+9,12*256+1,12*256+2,-1,-1])
        self.assertEqual(b['cu_seqlens_k'][:5].tolist(),[0,257,267,270,272])
        self.assertEqual(b['block_tables'][:4].tolist(),[[7,9,100,100],[11,100,100,100],[12,100,100,100],[100]*4])

    def test_shrinking_batch_resets_old_real_rows_to_scratch(self):
        r=self.runner()
        seqs=[NS(num_cached_tokens=255,block_table=[i*2,i*2+1]) for i in range(8)]
        r.prepare(seqs,[list(range(257)) for _ in seqs])
        small=[NS(num_cached_tokens=1,block_table=[20+i]) for i in range(3)]
        r.prepare(small,[[1,2,3]]*3)
        self.assertEqual(r.catchup_graph_vars['block_tables'][:4].tolist(),
                         [[20,100,100,100],[21,100,100,100],[22,100,100,100],[100]*4])
        self.assertEqual(r.catchup_graph_vars['slot_mapping'][6:8].tolist(),[-1,-1])
        self.assertEqual(r.catchup_graph_vars['input_ids'][6:8].tolist(),[0,0])

    def test_replay_projects_only_last_query_position_of_real_rows(self):
        reset=Mock();r=self.runner();r.catchup_graph_vars['outputs']=Buffer([[i] for i in range(8)])
        r.prepare_catchup_graph_inputs=Mock(return_value=4)
        r._set_catchup_graph_context=Mock();g=Mock();r.catchup_graphs={4:g}
        r._greedy_predictions=lambda hidden:[row[0]+100 for row in hidden.tolist()]
        replay=methods('replay_catchup_graph',reset_context=reset)['replay_catchup_graph']
        self.assertEqual(replay(r,[object()]*3,[[1,2]]*3),[[101],[103],[105]])
        g.replay.assert_called_once();reset.assert_called_once()

    def test_bad_query_length_rejected_before_any_buffer_mutation(self):
        r=self.runner()
        with self.assertRaises(ValueError):r.prepare([NS(num_cached_tokens=1,block_table=[0])],[[1,2]])
        self.assertEqual(r.catchup_graph_vars['input_ids'][0].tolist(),[999])


class TargetGraphTests(unittest.TestCase):
    def test_verify_q3_cross_page_and_shrinking_bucket(self):
        buffers = {name: Buffer([999] * 24) for name in ('input_ids', 'positions', 'slot_mapping')}
        buffers['cu_seqlens_k'] = Buffer([999] * 9)
        buffers['block_tables'] = Buffer([[999] * 4 for _ in range(8)])
        r = NS(verify_graph_bs=[1, 2, 4, 8], verify_query_len=3, verify_scratch_blocks=1,
               verify_graph_vars=buffers, config=NS(num_kvcache_blocks=100), block_size=256)
        torch = NS(tensor=lambda values, **kw: values, int32='i32', int64='i64')
        prepare = methods('prepare_verify_graph_inputs', torch=torch)['prepare_verify_graph_inputs']
        rows = [NS(num_cached_tokens=255, token_ids=list(range(256)), draft_tokens=[31, 32],
                   block_table=[2 * i, 2 * i + 1]) for i in range(8)]
        self.assertEqual(prepare(r, rows), 8)
        self.assertEqual(prepare(r, rows[:3]), 4)
        self.assertEqual(buffers['positions'][:12].tolist(), [255, 256, 257] * 3 + [0] * 3)
        self.assertEqual(buffers['slot_mapping'][:3].tolist(), [255, 256, 257])
        self.assertEqual(buffers['slot_mapping'][9:12].tolist(), [-1] * 3)
        self.assertEqual(buffers['block_tables'][3].tolist(), [[100] * 4])
        self.assertEqual(buffers['cu_seqlens_k'][:5].tolist(), [0, 258, 516, 774, 777])

    def test_target_decode_padding_uses_scratch_and_valid_length(self):
        buffers = {name: Buffer([999] * 16) for name in ('input_ids', 'positions', 'slot_mapping', 'context_lens')}
        buffers['block_tables'] = Buffer([[999] * 4 for _ in range(16)])
        r = NS(graph_bs=[1, 2, 4, 8, 16], graph_vars=buffers,
               config=NS(num_kvcache_blocks=100), block_size=256)
        torch = NS(tensor=lambda values, **kw: values, int32='i32', int64='i64')
        prepare = methods('prepare_decode', torch=torch)['prepare_decode']
        rows = [NS(block_table=[i]) for i in range(9)]
        self.assertEqual(prepare(r, rows, [[5, 7]] * 9), 16)
        self.assertEqual(buffers['slot_mapping'][9:16].tolist(), [-1] * 7)
        self.assertEqual(buffers['context_lens'][9:16].tolist(), [1] * 7)
        self.assertEqual(buffers['block_tables'][9:16].tolist(), [[100] * 4] * 7)


def runner_methods(*names):
    tree = ast.parse((ROOT / 'nanovllm/engine/model_runner.py').read_text(encoding='utf-8'))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    scope = {'defaultdict': defaultdict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual runner>', 'exec'), scope)
    return scope


class RoutingTests(unittest.TestCase):
    def test_capture_includes_target_decode_and_each_verify_query_length(self):
        scope = runner_methods('capture_graphs')
        captured = []
        def executor(runner, phase, q, cap):
            captured.append((phase, q, cap))
            return object()
        scope['capture_graphs'].__globals__['GraphExecutor'] = executor
        r = NS(config=NS(enforce_eager=False, name='target', max_num_seqs=16,
                         max_num_batched_tokens=4096, max_model_len=4096), graphs={})
        scope['capture_graphs'](r)
        self.assertEqual(captured, [('decode', 1, 128), ('verify', 2, 16),
                                   ('verify', 3, 8), ('verify', 4, 4), ('verify', 5, 2)])

    def test_next_tokens_groups_1_2_and_eager_without_reordering(self):
        calls = []
        class Graph:
            def __init__(self, phase):
                self.phase = phase
                self.config = NS(max_num_seqs=16)
            def replay(self, rows, tokens):
                calls.append(self.phase)
                return [[row.request_id] for row in rows]
        r = NS(graphs={('decode', 1): Graph('decode'), ('catchup', 2): Graph('catchup')}, stats=Counter())
        def eager(rows, tokens):
            calls.append('eager')
            return [[row.request_id] for row in rows]
        r._predict = eager
        rows = [NS(request_id=i, num_cached_tokens=cached) for i, cached in enumerate([4, 3, 0, 4])]
        result = runner_methods('next_tokens')['next_tokens'](r, rows, [[1] * 5] * 4)
        self.assertEqual(result, [0, 1, 2, 3])
        self.assertEqual(calls, ['decode', 'catchup', 'eager'])

    def test_verify_matches_q_and_batch_overflow_falls_back(self):
        calls = []
        class Graph:
            def __init__(self, q, cap):
                self.q = q
                self.config = NS(max_num_seqs=cap)
            def replay(self, rows):
                calls.append(('graph', self.q))
                return [[row.request_id] * self.q for row in rows]
        r = NS(graphs={('verify', 2): Graph(2, 1), ('verify', 5): Graph(5, 2)}, stats=Counter())
        def eager(rows, tokens, all_positions):
            calls.append(('eager', len(rows)))
            return [[row.request_id] * (len(row.draft_tokens) + 1) for row in rows]
        r._predict = eager
        rows = [NS(request_id=i, token_ids=[1, 2], num_cached_tokens=1, draft_tokens=[9] * m)
                for i, m in enumerate([4, 1, 1, 2])]
        result = runner_methods('verify')['verify'](r, rows)
        self.assertEqual(result, [[0] * 5, [1] * 2, [2] * 2, [3] * 3])
        self.assertEqual(calls, [('graph', 5), ('eager', 2), ('eager', 1)])

