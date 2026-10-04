"""Prevent scheduler identifiers from appearing as completed training steps."""
import unittest
from unittest.mock import patch

from dashboard import families, model
from dashboard.sources.runs import _parse_state


class TrainingProgressTest(unittest.TestCase):
    def test_slurm_step_cannot_advance_training_timeline(self):
        execution = _parse_state('RUNNING 2026-09-12T19:03:16+00:00 job=5926 step=577 host=node')
        row = {'id': 'run', 'path': 'logs/run', 'state': 'RUNNING', 'exec': execution,
               'progress': {}, 'checkpoint_steps': [], 'validation_steps': [0], 'exports': []}
        variant = {'id': 'V0096', 'parameters': {'steps': 2000}}
        with patch.object(model.runs, 'scan', return_value={}):
            before = model.timeline(variant, [row], [], {})
            self.assertEqual(before['segments'][0]['end'], 0)
            self.assertEqual([p['step'] for p in before['datapoints']], [0])
            row['progress'] = {'step': 12}
            after = model.timeline(variant, [row], [], {})
            self.assertEqual(after['segments'][0]['end'], 12)
            self.assertEqual([p['step'] for p in after['datapoints']], [0])

    def test_versioned_tiny_subset_has_its_own_lane(self):
        params = {'attention_backend': 'VSQA', 'dimensions': 'quality-7',
                  'prompt_set': 'q7tiny.v1.json (pinned 32 prompts)'}
        tiny, error = families.classify_eval(params)
        full, _ = families.classify_eval({**params, 'prompt_set': 'VBench_full_info.json'})
        self.assertIsNone(error)
        self.assertEqual(tiny, 'vbench[q7tiny-v1]')
        self.assertNotEqual(tiny, full)

    def test_dataset_is_the_only_vbench_group(self):
        dataset = 'q7tiny.v1.json (pinned 32 prompts)'
        sparse = {'attention_backend': 'VSQA (FVFA4-v3 NVFP4-QK/FP8-PV, cube 8x4x8)', 'attention_kind': 'vsa',
                  'dimensions': 'quality-7', 'prompt_set': dataset}
        dense_qat = {'attention_backend': 'ATTN_QAT_FP8_PV_TRAIN', 'attention_kind': 'dense',
                     'dimensions': 'quality-7', 'prompt_set': dataset}
        bf16 = {'attention_backend': 'TORCH_SDPA', 'attention_kind': 'dense',
                'dimensions': 'quality-7', 'prompt_set': dataset}
        kinds = [families.classify_eval(p) for p in (sparse, dense_qat, bf16)]
        self.assertEqual([k for k, _ in kinds], ['vbench[q7tiny-v1]'] * 3, 'any suitable forward shares the dataset lane')
        self.assertEqual([e for _, e in kinds], [None] * 3)

    def test_two_forwards_of_one_checkpoint_stay_two_datapoints(self):
        variant = {'id': 'V0126', 'parameters': {'steps': 2000, 'attention_kind': 'dense', 'cube_shape': 'none',
                                                 'model': 'wan-t2v-1.3b', 'backend': 'ATTN_QAT_FP8_PV_TRAIN'}}
        base = {'step': 2000, 'status': 'COMPLETED', 'kind': 'vbench[q7tiny-v1]', 'kind_error': None,
                'quality_score': 0.5, 'metrics': {}, 'prompts': 32, 'runs': []}
        evaluations = [{**base, 'id': 'V0081', 'backend': 'dense BF16', 'attention_backend': 'TORCH_SDPA'},
                       {**base, 'id': 'V0085', 'backend': 'NVFP4/FP8·dense', 'attention_backend': 'ATTN_QAT_FP8_PV_TRAIN'}]
        dps = model.timeline(variant, [], evaluations, {'inf_val': {}, 'vbench': {}})['datapoints']
        lane = [d for d in dps if d['kind'] == 'vbench[q7tiny-v1]' and d['step'] == 2000]
        self.assertEqual([d['eval'] for d in lane], ['V0081', 'V0085'])

    def test_unknown_evaluation_does_not_break_other_datapoints(self):
        variant = {'id': 'V0053', 'parameters': {'steps': 1000}}
        kind, error = families.classify_eval({'attention_backend': 'unknown', 'prompt_set': 'unregistered'})
        self.assertIsNone(kind)
        base = {'step': 1000, 'status': 'PLANNED', 'quality_score': None, 'backend': 'dense BF16',
                'attention_backend': None, 'metrics': {}, 'prompts': 32, 'runs': []}
        evaluations = [{**base, 'id': 'V0001', 'kind': 'vbench[q7]', 'kind_error': None},
                       {**base, 'id': 'V0002', 'kind': kind, 'kind_error': error}]
        actual = model.timeline(variant, [], evaluations, {'inf_val': {}, 'vbench': {}})['datapoints']
        self.assertEqual({d['eval'] for d in actual}, {'V0001', 'V0002'})
        self.assertEqual(next(d for d in actual if d['eval'] == 'V0002')['error'], error)

    def test_structured_precision_overrides_legacy_description(self):
        current = families.fam_kernel({'backend': 'ATTN_QAT_FP8_PV_TRAIN',
                                       'pv_scaling': 'none', 'quantization': 'PV scale trick'})
        self.assertFalse(current['expanded']['pv_scale'])
        scaled = families.fam_kernel({'backend': 'ATTN_QAT_FP8_PV_TRAIN',
                                      'pv_scaling': 'per_tile', 'quantization': 'no PV scaling'})
        self.assertTrue(scaled['expanded']['pv_scale'])


if __name__ == '__main__':
    unittest.main()
