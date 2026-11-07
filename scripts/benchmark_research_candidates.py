"""Matched cached CPU latency; excludes loading, file checks and sensor acquisition."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from cranepilot.benchmark import environment_record
from cranepilot.broad_strategies import make_model
from cranepilot.physics import LengthCalibration
from cranepilot.forced_observer import refine_windows, analytic_rollout
from scripts.run_temporal import load_actor
from scripts.run_broad_search import inputs, predict
from scripts.predict_cranepilot import build_packet
from scripts.reliability_blend import blend


def main():
    dest = Path('results/broad_search_20261107/cached_cpu_latency.json')
    if dest.exists(): raise FileExistsError('Preserve earlier benchmark')
    run = Path('artifacts/runs/broad_search_20261107')
    p = json.loads((run / 'provenance.json').read_text()); cfg = p['config']; info = p['inputs']['primary']
    actor = load_actor(info); cal = LengthCalibration(**info['calibration']); torch.set_num_threads(4)
    with np.load('examples/history_packet.npz') as z:
        packet = {k: z[k][:1].copy() for k in z.files}
    ensembles = {}
    for name in ['incumbent', 'soft_symmetry']:
        models = []
        for i, seed in enumerate(cfg['seeds']):
            model = make_model({} if name == 'incumbent' else cfg['variants'][name])
            path = Path(info['weights'][i]) if name == 'incumbent' else run / 'primary' / f'{name}_seed{seed}.pt'
            model.load_state_dict(torch.load(path, weights_only=True)); models.append(model.eval())
        ensembles[name] = models

    def invoke(name):
        raw = build_packet(packet['theta'], packet['x'], packet['reference'], packet['length_m'], packet['future_reference'], cal)
        batch = inputs(raw, actor)
        ensemble = ensembles['incumbent' if name == 'gate_0.5' else name]
        result = np.mean([predict(model, batch) for model in ensemble], axis=0)
        if name == 'gate_0.5':
            refined = refine_windows(raw)
            physical = analytic_rollout(refined['initial_q'], refined['initial_p'], raw['omega'], batch['acceleration'].numpy())
            rms = np.sqrt(np.mean(refined['forced_residual']**2, axis=1))
            result, _ = blend(result, physical, rms, .5)
        return result

    names = ['incumbent', 'soft_symmetry', 'gate_0.5']
    for _ in range(5):
        for name in names: invoke(name)
    samples = {n: [] for n in names}
    # Rotate the measurement order to reduce a simple warmup/order confound.
    for i in range(50):
        for name in names[i % 3:] + names[:i % 3]:
            start = time.perf_counter(); y = invoke(name); elapsed = time.perf_counter() - start
            if not np.isfinite(y).all(): raise AssertionError('Nonfinite forecast')
            samples[name].append(elapsed * 1000)
    report = {'scope': 'One cached CPU packet, causal preprocessing + shared frozen actor + three angular seeds; no disk IO/hash checks/loading/sensor time. No realtime guarantee.',
              'torch_threads': 4, 'warmup_each': 5, 'timed_each': 50, 'environment': environment_record(),
              'results': {n: {'median_ms': float(np.median(x)), 'p95_ms': float(np.quantile(x, .95)), 'samples_ms': x} for n, x in samples.items()}}
    dest.parent.mkdir(exist_ok=True, parents=True); dest.write_text(json.dumps(report, indent=2))
    print(json.dumps({n: {k: v for k, v in x.items() if k != 'samples_ms'} for n, x in report['results'].items()}, indent=2))


if __name__ == '__main__': main()
