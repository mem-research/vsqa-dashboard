"""Training-stage graph and explicit checkpoint branches; planning never launches a Run."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .config import TRAIN_EXP, PROJECT_ROOT
from .sources import results, runs


def checkpoints(v: dict, run_rows: list[dict]) -> list[dict]:
    """Keep physical Run identity: equal step numbers from different Runs are distinct."""
    out = []
    for r in run_rows:
        directory = runs.run_dir(r['path'])
        exports = {(e['step'], bool(e['ema'])): e for e in r.get('exports') or [] if e['has_metadata']}
        for step in sorted(set(r.get('checkpoint_steps') or []) | {s for s, _ in exports}):
            raw = directory / 'checkpoints' / f'checkpoint-{step}'
            for ema in (False, True):
                export = exports.get((step, ema))
                if not export and (ema or not (raw / 'metadata.json').is_file()):
                    continue
                artifact = export['path'] if export else str(raw.relative_to(PROJECT_ROOT))
                out.append({'variant': v['id'], 'step': step, 'ema': ema, 'run': r['path'],
                            'artifact': artifact, 'available': True, 'exported': bool(export)})
    # A declared target can be used to design the next stage before training finishes.
    target = (v.get('parameters') or {}).get('steps')
    if isinstance(target, int) and target > 0 and not any(c['step'] == target and not c['ema'] for c in out):
        out.append({'variant': v['id'], 'step': target, 'ema': False, 'run': None,
                    'artifact': None, 'available': False, 'exported': False})
    return out


def graph() -> dict:
    from . import model, families
    overview = {v['id']: v for v in model.overview()}
    registry = {v['id']: v for v in results.variants(TRAIN_EXP)}
    nodes, edges = {}, []
    pending = list(dict.fromkeys([*overview, *(vid for vid, v in registry.items()
                                              if (v.get('provenance') or {}).get('checkpoint_parent'))]))
    while pending:
        vid = pending.pop(0)
        if vid in nodes:
            continue
        v = registry.get(vid)
        if v is None:
            nodes[vid] = {'id': vid, 'name': vid, 'external': True, 'status': 'UNKNOWN', 'checkpoints': []}
            continue
        p = v.get('parameters') or {}
        row = overview.get(vid)
        run_rows = row['run_rows'] if row else model.variant_runs(v, model._wandb_url_index())
        source = model.resume_source(v)
        plan_start = p.get('resume_step') or 0
        observed = max([r.get('progress', {}).get('step', 0) or 0 for r in run_rows] +
                       [s for r in run_rows for s in (r.get('checkpoint_steps') or []) + (r.get('validation_steps') or [])] + [plan_start])
        nodes[vid] = {'id': vid, 'name': v.get('name'), 'status': v.get('status'), 'external': False,
                      'method': families.fam_objective(p)['expanded']['objective'] or p.get('objective'),
                      'start': plan_start, 'target': p.get('steps'), 'observed': observed,
                      'has_execution': bool(run_rows), 'checkpoints': checkpoints(v, run_rows),
                      'runs': [r['path'] for r in run_rows], 'warnings': row['warnings'] if row else [],
                      'source': source}
        if source['kind'] == 'variant':
            parent = source['variant']
            pending.append(parent)
        else:
            label = source.get('label') or 'Unspecified source'
            parent = 'source-' + hashlib.sha256(label.encode()).hexdigest()[:16]
            nodes.setdefault(parent, {'id': parent, 'name': label, 'external': True,
                                      'status': 'BASE' if source['kind'] == 'base' else 'EXTERNAL', 'checkpoints': []})
        relation = ((v.get('provenance') or {}).get('checkpoint_parent') or {}).get('relation')
        edges.append({'from': parent, 'to': vid, 'step': source.get('step'), 'ema': source.get('ema', False),
                      'run': source.get('run'), 'kind': relation or
                      ('base weights' if source['kind'] == 'base' else 'checkpoint source (resume/inheritance unverified)')})
    # Surface historical cycles instead of hiding nodes or hanging layout.
    remaining = set(nodes)
    layers = {}
    while remaining:
        ready = sorted(n for n in remaining if all(e['from'] not in remaining for e in edges if e['to'] == n))
        if not ready:
            break
        for n in ready:
            layers[n] = max([layers[e['from']] + 1 for e in edges if e['to'] == n] + [0])
        remaining.difference_update(ready)
    for n in nodes:
        nodes[n]['layer'] = layers.get(n, max(layers.values(), default=0) + 1)
    return {'nodes': list(nodes.values()), 'edges': edges,
            'warnings': ['Cyclic lineage: ' + ', '.join(sorted(remaining))] if remaining else []}


def build_branch(form: dict) -> dict:
    from . import model, variants
    source = form['source']
    parent = results.get(TRAIN_EXP, source['variant'])
    if not parent:
        raise ValueError('Parent Variant no longer exists')
    choices = checkpoints(parent, model.variant_runs(parent, model._wandb_url_index()))
    matches = [c for c in choices if c['step'] == source['step'] and c['ema'] == source.get('ema', False)
               and c['run'] == source.get('run')]
    if len(matches) != 1:
        raise ValueError('Checkpoint is missing or ambiguous; refresh the graph and select it again')
    checkpoint = matches[0]
    if not results.get(TRAIN_EXP, form.get('clone_from') or ''):
        raise ValueError('Select an existing training template')
    # Use exactly the reviewed recipe; never retain hidden template environment values.
    row = variants.build_train_variant({**form, 'clone_from': None})
    params = row['parameters']
    if params.get('objective') not in ('fine_tune', 'dense_qat', 'dmd', 'dmd2'):
        raise ValueError('Unsupported training objective')
    if not isinstance(params.get('steps'), int) or isinstance(params['steps'], bool) or params['steps'] <= 0:
        raise ValueError('steps must be a positive integer')
    if not row['provenance'].get('entry'):
        raise ValueError('Launcher entry is required')
    entry = Path(row['provenance']['entry'])
    if entry.is_absolute() or '..' in entry.parts or not (PROJECT_ROOT / entry).is_file() or not (PROJECT_ROOT / entry).resolve().is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError('Launcher must be an existing project-relative file')
    # A branch is a weight-only new stage, not a DCP optimizer-state resume.
    params['resume_step'] = 0
    for key in ('init_step', 'dense_qat_step', 'sparse_ft_step'):
        params.pop(key, None)
    params['initializer'] = checkpoint['artifact'] or f"E0029/{checkpoint['variant']}@{checkpoint['step']}"
    env = row['provenance']['env']
    for key in ('RUN_NAME', 'ALLOCATION_ID', 'RESUME', 'RESUME_FROM', 'RESUME_FROM_CHECKPOINT', 'INITIALIZER'):
        env.pop(key, None)
    env['MAX_STEPS'] = str(params['steps'])
    if checkpoint['exported']:
        env['INITIALIZER'] = checkpoint['artifact']
    row['provenance']['checkpoint_parent'] = {**checkpoint, 'relation': 'weight_initialization'}
    row['provenance']['notes'] = (row['provenance'].get('notes', '') +
        '\nDAG branch: weight-only initialization, not optimizer resume. ' +
        ('Source export available.' if checkpoint['exported'] else
         'Blocked on source checkpoint/export preparation; no runnable initializer is claimed.') +
        ' Training launch requires recipe/metadata preflight; this operation only declares a stage.')
    return row
