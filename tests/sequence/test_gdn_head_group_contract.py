"""CPU checks: python3 -m unittest tests.sequence.test_gdn_head_group_contract."""
from dataclasses import replace
import unittest

from b12x.preparation import FrozenMapping
from b12x.sequence.gdn_decode._tuning import GdnConfig, GdnQuery, TUNING


class HeadGroupContract(unittest.TestCase):
    def setUp(self):
        self.query = GdnQuery(
            gate_activation="sigmoid", qk_l2norm=True, state_dtype="float32",
            key_heads=8, value_heads=24, max_seqs=16, max_tokens=80,
            state_index_columns=5,
        )

    def test_candidates_round_trip_and_keep_legacy_default(self):
        self.assertEqual(TUNING.default_config(self.query, None).qwen_head_group_size, 0)
        configs = [config for _, config in TUNING.choices(self.query, None)]
        self.assertEqual({config.qwen_head_group_size for config in configs}, {0})
        for group in (0, 1, 2, 3):
            config = GdnConfig(backend="cutedsl", recurrent_block_v=32, qwen_head_group_size=group)
            TUNING.validate_config(self.query, config, None)
            self.assertEqual(TUNING.decode_config(TUNING.config_payload(config)), config)

    def test_kda_does_not_acquire_qwen_candidates(self):
        query = replace(self.query, value_heads=8)
        self.assertEqual({config.qwen_head_group_size for _, config in TUNING.choices(query, None)}, {0})
        with self.assertRaises(ValueError):
            TUNING.validate_config(query, GdnConfig(backend="triton", recurrent_block_v=32, qwen_head_group_size=2), None)

    def test_invalid_groups_and_stale_serialized_configs_fail_closed(self):
        for group in (True, 1.5, -1, 4):
            with self.subTest(group=group), self.assertRaises((TypeError, ValueError)):
                TUNING.validate_config(self.query, GdnConfig(backend="cutedsl", recurrent_block_v=32, qwen_head_group_size=group), None)
        with self.assertRaises(ValueError):
            TUNING.decode_config(FrozenMapping({"backend": "cutedsl", "recurrent_block_v": 32}))


if __name__ == "__main__":
    unittest.main()
