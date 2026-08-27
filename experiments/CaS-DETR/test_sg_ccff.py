import unittest

import torch

from engine.deim.hybrid_encoder import HybridEncoder


class SGCCFFTest(unittest.TestCase):
    def test_small_forward(self):
        encoder = HybridEncoder(
            in_channels=[16, 16, 16], hidden_dim=16, nhead=4,
            dim_feedforward=32, use_encoder_idx=[2], depth_mult=0.34,
            expansion=0.5, token_keep_ratio=0.5,
            enable_cas_predictor=True, use_cass=True, use_dynamic=False,
            enable_sg_ccff=True,
        ).eval()
        feats = [
            torch.randn(1, 16, 8, 8),
            torch.randn(1, 16, 4, 4),
            torch.randn(1, 16, 2, 2),
        ]

        with torch.no_grad():
            outputs, info = encoder(feats, return_encoder_info=True)

        self.assertEqual([tuple(x.shape) for x in outputs], [
            (1, 16, 8, 8), (1, 16, 4, 4), (1, 16, 2, 2)
        ])
        self.assertEqual(encoder.sg_ccff_alpha.numel(), 2)
        self.assertTrue(torch.equal(encoder.sg_ccff_alpha, torch.zeros(2)))
        self.assertGreater(float(info['token_pruning_ratios'][0]), 0.0)


if __name__ == '__main__':
    unittest.main()
