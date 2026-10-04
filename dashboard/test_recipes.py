"""Checkpoint identity and planned lineage must survive ambiguous Run/step labels."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from dashboard import model, recipes
from dashboard.config import TRAIN_EXP
from dashboard.sources import results


class RecipeBranchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'launch.sh').write_text('#!/bin/sh\n')
        self.rows = []
        for run in ('first', 'second'):
            raw = self.root / 'logs' / run / 'checkpoints' / 'checkpoint-50'
            raw.mkdir(parents=True)
            (raw / 'metadata.json').write_text('{}')
            self.rows.append({'path': f'logs/{run}', 'checkpoint_steps': [50], 'exports': []})
        self.parent = {'id': 'V0001', 'name': 'parent', 'status': 'PLANNED',
                       'parameters': {'steps': 100, 'objective': 'fine_tune'}, 'runs': []}
        # No trailing newline exercises the preserving textual append path.
        self.path = self.root / 'results.yaml'
        self.original = yaml.safe_dump({'columns': [], 'variants': [self.parent]}, sort_keys=False).rstrip('\n')
        self.path.write_text(self.original)
        results._cache.pop(TRAIN_EXP, None)
        self.addCleanup(results._cache.pop, TRAIN_EXP, None)
        for context in (patch.dict(results.EXP_DIR, {TRAIN_EXP: self.root}),
                        patch.object(recipes, 'PROJECT_ROOT', self.root),
                        patch.object(recipes.runs, 'run_dir', side_effect=lambda path: self.root / path),
                        patch.object(model, 'variant_runs', side_effect=lambda v, *args: self.rows if v['id'] == 'V0001' else []),
                        patch.object(model, '_wandb_url_index', return_value={})):
            context.start()
            self.addCleanup(context.stop)

    def form(self, step=50, run='logs/second'):
        return {'source': {'variant': 'V0001', 'step': step, 'run': run},
                'clone_from': 'V0001', 'name': 'smoke-named planned stage', 'entry': 'launch.sh',
                'parameters': {'objective': 'fine_tune', 'steps': 150}, 'env': {}}

    def test_equal_steps_from_distinct_runs_do_not_alias(self):
        row = recipes.build_branch(self.form())
        self.assertEqual(row['parameters']['initializer'], 'logs/second/checkpoints/checkpoint-50')
        self.assertNotIn('INITIALIZER', row['provenance']['env'])  # Raw DCP is not a runnable export.
        with self.assertRaisesRegex(ValueError, 'missing or ambiguous'):
            recipes.build_branch(self.form(run=None))

    def test_future_branch_persists_without_execution_and_remains_in_graph(self):
        row = recipes.build_branch(self.form(step=100, run=None))
        vid = results.append_variant(TRAIN_EXP, row)
        self.assertTrue(self.path.read_text().startswith(self.original + '\n'))
        results._cache.pop(TRAIN_EXP, None)
        loaded = results.get(TRAIN_EXP, vid)
        self.assertEqual(loaded['status'], 'PLANNED')
        self.assertEqual(loaded['runs'], [])
        self.assertEqual(model.parent_of(loaded)['variant'], 'V0001')
        self.assertNotIn('INITIALIZER', loaded['provenance']['env'])
        with patch.object(model, 'overview', return_value=[]):
            graph = recipes.graph()
        self.assertIn(vid, {node['id'] for node in graph['nodes']})
        edge = next(edge for edge in graph['edges'] if edge['to'] == vid)
        self.assertEqual((edge['from'], edge['step']), ('V0001', 100))


if __name__ == '__main__':
    unittest.main()
