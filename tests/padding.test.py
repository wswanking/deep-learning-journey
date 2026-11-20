"""Exercise chapter-22 padding semantics and deliberately broken negative controls."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/22-instruction-tuning'))
sys.path.insert(1, str(ROOT / 'examples/21-tiny-language-model'))
import torch
from torch.nn import functional as F
from run import check_right_padding
from model import load_checkpoint
from sft_data import load_splits, collate


class PaddingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model, _ = load_checkpoint(ROOT / 'build/experiments/21-tiny-language-model/base-model.pt')
        splits, _ = load_splits()
        cls.batch = collate(splits['train'][:8], cls.model.config.context_length)

    def test_trained_model_passes_both_precisions(self):
        result = check_right_padding(self.model, self.batch)['right_padding_checks']
        self.assertLess(result['float64']['max_logit_difference'], 1e-11)
        self.assertTrue(result['float32']['padding_content_bitexact'])

    def test_missing_combined_causal_and_padding_mask_is_rejected(self):
        # Removes BOTH causal and padding constraints, not just key padding.
        attention = F.scaled_dot_product_attention
        def unmasked(*args, **kwargs):
            kwargs['attn_mask'] = None
            return attention(*args, **kwargs)
        with patch.object(F, 'scaled_dot_product_attention', unmasked):
            with self.assertRaisesRegex(AssertionError, 'padding changed|padding content leaked'):
                check_right_padding(copy.deepcopy(self.model), self.batch)

    def test_float32_only_batch_corruption_is_rejected(self):
        broken = copy.deepcopy(self.model)
        def corrupt(module, inputs, output):
            if output.dtype == torch.float32 and output.shape[0] > 1:
                return output + 1e-3
            return output
        handle = broken.register_forward_hook(corrupt)
        try:
            with self.assertRaisesRegex(AssertionError, 'padding changed.*float32'):
                check_right_padding(broken, self.batch)
        finally:
            handle.remove()


if __name__ == '__main__':
    unittest.main()
