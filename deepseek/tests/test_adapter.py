import unittest

import torch

from deepseek.sglang_adapter import pack_sparse_prefill


class AdapterTests(unittest.TestCase):
    def test_packing_rebases_batches_and_compacts_valid_prefixes(self):
        local = torch.arange(12.0).reshape(2, 3, 2)
        main = torch.arange(8.0).reshape(2, 2, 2) + 100
        local_ids = torch.tensor([[[-1, 0, 1], [1, 2, -1]]], dtype=torch.int32).expand(
            2, -1, -1
        )
        main_ids = torch.tensor([[[0, -1], [-1, 1]]], dtype=torch.int32).expand(
            2, -1, -1
        )
        kv, ids, lengths = pack_sparse_prefill(local, main, local_ids, main_ids)
        self.assertEqual(ids.shape, (4, 1, 64))
        self.assertEqual(lengths.tolist(), [3, 3, 3, 3])
        self.assertEqual(
            ids[:, 0, :3].tolist(), [[0, 1, 3], [1, 2, 4], [5, 6, 8], [6, 7, 9]]
        )
        self.assertTrue((ids[:, 0, 3:] == -1).all())
        for b in range(2):
            for t in range(2):
                expected = torch.cat(
                    (
                        local[b, local_ids[b, t][local_ids[b, t] >= 0].long()],
                        main[b, main_ids[b, t][main_ids[b, t] >= 0].long()],
                    )
                )
                torch.testing.assert_close(kv[ids[b * 2 + t, 0, :3].long()], expected)
