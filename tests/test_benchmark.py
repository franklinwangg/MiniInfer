import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from miniinfer.benchmark import measure_run
from miniinfer.generation import greedy_generate
from miniinfer.model import from_reference


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        config = Qwen2Config(
            vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
            tie_word_embeddings=True,
        )
        cls.model = from_reference(Qwen2ForCausalLM(config).eval())
        cls.prompt = torch.tensor([[2, 17, 8, 31, 4]])

    def test_measurement_matches_generation(self):
        metrics, tokens = measure_run(self.model, self.prompt, 4)
        expected = greedy_generate(self.model, self.prompt, 4)[:, self.prompt.shape[1]:]
        self.assertTrue(torch.equal(tokens, expected))
        self.assertEqual(metrics["decode_tokens"], 3)
        self.assertGreater(metrics["prefill_seconds"], 0)
        self.assertGreater(metrics["decode_seconds"], 0)
        self.assertAlmostEqual(
            metrics["overall_tokens_per_second"], 4 / metrics["total_seconds"]
        )
        self.assertAlmostEqual(
            metrics["decode_tokens_per_second"], 3 / metrics["decode_seconds"]
        )
        self.assertIsNone(metrics["peak_cuda_allocated_bytes"])

    def test_invalid_workloads(self):
        with self.assertRaises(ValueError):
            measure_run(self.model, self.prompt, 1)
        with self.assertRaises(ValueError):
            measure_run(self.model, self.prompt, 28)
        with self.assertRaises(ValueError):
            measure_run(self.model, torch.empty((1, 0), dtype=torch.long), 2)


if __name__ == "__main__":
    unittest.main()
