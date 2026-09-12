import unittest

import torch

from deepseek.config import global_cache_bytes, index_positions_per_decode, layer_spec
from deepseek.csa2_reference import (
    CSA2Router,
    StreamingCompressor,
    causal_index_scores,
    gated_compress,
    index_scores,
    joint_sparse_attention,
    reindex_candidates,
    select_candidates,
    select_topk,
    window_indices,
)


class CSA2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_released_schedule_and_cache_accounting(self):
        modes = [layer_spec(i).mode for i in range(40)]
        self.assertEqual(
            {m: modes.count(m) for m in set(modes)},
            {"SWA": 2, "Full": 4, "Reindex": 4, "Reuse": 30},
        )
        self.assertEqual(layer_spec(39).kv_owner, 20)
        self.assertEqual(layer_spec(39).index_owner, 36)
        self.assertEqual(global_cache_bytes(1048576), 890 * 1048576)
        self.assertEqual(global_cache_bytes(3), 6 * 356)
        self.assertEqual(index_positions_per_decode(1048576), 2686976)

    def test_compression_is_channelwise_and_discards_partial_group(self):
        v = torch.tensor([[[1.0, 9.0], [7.0, 3.0], [100.0, 100.0]]])
        gates = torch.tensor([[[0.0, 40.0], [40.0, 0.0], [0.0, 0.0]]])
        torch.testing.assert_close(
            gated_compress(v, gates, 2), torch.tensor([[[7.0, 9.0]]])
        )
        torch.testing.assert_close(
            gated_compress(v, torch.zeros_like(v), 2), torch.tensor([[[4.0, 6.0]]])
        )
        self.assertIs(gated_compress(v, None, 1), v)

    def test_streaming_matches_prefill_across_odd_boundaries(self):
        for ratio in (1, 2):
            v, g = torch.randn(2, 13, 8), torch.randn(2, 13, 8)
            for chunks in ([1] * 13, [3, 4, 1, 5], [13]):
                stream = StreamingCompressor(ratio)
                emitted, start = [], 0
                for n in chunks:
                    emitted.append(
                        stream.append(v[:, start : start + n], g[:, start : start + n])
                    )
                    start += n
                torch.testing.assert_close(
                    torch.cat(emitted, 1), gated_compress(v, g, ratio)
                )
                self.assertEqual(stream.tokens_seen, 13)

    def test_completed_group_causality_and_prefix_invariance(self):
        q, keys, w = torch.randn(1, 7, 2, 4), torch.randn(1, 3, 4), torch.randn(1, 7, 2)
        scores = causal_index_scores(q, keys, w, torch.arange(7), 2)
        for t in range(7):
            n = (t + 1) // 2
            self.assertEqual(torch.isfinite(scores[0, t]).sum().item(), n)
            prefix = causal_index_scores(
                q[:, t : t + 1], keys[:, :n], w[:, t : t + 1], torch.tensor([t]), 2
            )
            torch.testing.assert_close(
                select_topk(prefix, 5), select_topk(scores[:, t : t + 1], 5)
            )
        keys[:, 2] = 1e6
        changed = causal_index_scores(q, keys, w, torch.arange(7), 2)
        torch.testing.assert_close(scores[:, :5], changed[:, :5])

    def test_newest_candidate_block_is_pinned_and_empty_rows_are_safe(self):
        scores = torch.tensor([[[100.0, 99.0, 90.0, 80.0, -5.0, -torch.inf]]])
        ids = select_candidates(scores, torch.tensor([[5]]), 1, 2)
        self.assertEqual(ids.tolist(), [[[4, -1]]])
        empty = select_candidates(
            torch.full_like(scores, -torch.inf), torch.tensor([[0]]), 2, 2
        )
        self.assertTrue((empty == -1).all())
        short = select_topk(torch.tensor([[2.0, -torch.inf, 1.0]]), 5)
        self.assertEqual(short.tolist(), [[0, 2, -1, -1, -1]])

    def test_gather_first_reindex_equals_dense_masked_oracle(self):
        q, keys, w = torch.randn(2, 9, 3, 4), torch.randn(2, 9, 4), torch.randn(2, 9, 3)
        first_scores = causal_index_scores(q, keys, w, torch.arange(9), 1)
        candidates = select_candidates(first_scores, torch.arange(1, 10)[:, None], 2, 2)
        q2, w2 = torch.randn_like(q), torch.randn_like(w)
        full_scores = index_scores(q2, keys, w2)
        mask = torch.zeros_like(full_scores, dtype=torch.bool)
        for b in range(2):
            for t in range(9):
                ids = candidates[b, t]
                mask[b, t, ids[ids >= 0].long()] = True
        expected = select_topk(full_scores.masked_fill(~mask, -torch.inf), 6)
        actual = reindex_candidates(q2, keys, w2, candidates, 6)
        torch.testing.assert_close(actual, expected)

    def test_joint_attention_against_dense_masked_oracle(self):
        b, t, h, d = 2, 5, 3, 4
        q, local, main = (
            torch.randn(b, t, h, d),
            torch.randn(b, t, d),
            torch.randn(b, 2, d),
        )
        local_ids = window_indices(torch.arange(t), 3, b)
        main_ids = torch.tensor(
            [[[-1, -1], [0, -1], [0, -1], [0, 1], [0, 1]]], dtype=torch.int32
        ).expand(b, -1, -1)
        sink = torch.randn(h)
        actual = joint_sparse_attention(q, local, main, local_ids, main_ids, sink)
        all_kv = torch.cat((local, main), 1)
        scores = torch.einsum("bthd,bnd->bthn", q, all_kv) / d**0.5
        allowed = torch.zeros(b, t, t + 2, dtype=torch.bool)
        for batch in range(b):
            for pos in range(t):
                for ids, offset in ((local_ids, 0), (main_ids, t)):
                    valid = ids[batch, pos]
                    allowed[batch, pos, valid[valid >= 0].long() + offset] = True
        scores = scores.masked_fill(~allowed[:, :, None], -torch.inf)
        probs = torch.cat(
            (scores, sink[None, None, :, None].expand(b, t, h, 1)), -1
        ).softmax(-1)[..., :-1]
        expected = torch.einsum("bthn,bnd->bthd", probs, all_kv)
        torch.testing.assert_close(actual, expected)

    def test_sink_joint_normalization_and_invalid_slots(self):
        q = torch.zeros(1, 1, 1, 1)
        local, main = torch.tensor([[[2.0]]]), torch.tensor([[[4.0]]])
        ids = torch.zeros(1, 1, 1, dtype=torch.int32)
        out = joint_sparse_attention(q, local, main, ids, ids, torch.zeros(1))
        # Equal logits: (2 + 4 + zero-value sink) / 3, not two separate softmaxes.
        torch.testing.assert_close(out, torch.tensor([[[[2.0]]]]))
        invalid = torch.full_like(ids, -1)
        out = joint_sparse_attention(q, local, main, invalid, invalid, torch.zeros(1))
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.item(), 0.0)
        with self.assertRaises(ValueError):
            joint_sparse_attention(q, local, main, ids + 1, ids, torch.zeros(1))

    def test_router_cache_identity_reindex_ownership_and_lifetime(self):
        router = CSA2Router(torch.arange(9), topk=3, candidate_blocks=2, block_size=2)
        last = None
        for i in range(40):
            spec = layer_spec(i)
            kwargs = {}
            if spec.mode == "Full":
                n = 9 // spec.ratio
                kwargs.update(
                    main_kv=torch.randn(1, n, 8), index_k=torch.randn(1, n, 4)
                )
            if spec.mode in ("Full", "Reindex"):
                kwargs.update(q=torch.randn(1, 9, 2, 4), weights=torch.randn(1, 9, 2))
            current = router.route(i, **kwargs)
            if spec.mode == "Reuse":
                self.assertIs(current.main_kv, last.main_kv)
                self.assertIs(current.indices, last.indices)
            elif spec.mode == "Reindex":
                self.assertIs(current.main_kv, last.main_kv)
                self.assertIsNot(current.indices, last.indices)
                for t in range(9):
                    candidates = set(router.candidates[0, t].tolist())
                    self.assertTrue(
                        set(current.indices[0, t].tolist()) <= candidates | {-1}
                    )
            last = current
        with self.assertRaises(ValueError):
            router.route(0)


if __name__ == "__main__":
    unittest.main()
