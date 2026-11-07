"""Evaluate a fixed composition on retrospective records and new mismatch stress."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
from cranepilot.benchmark import environment_record
from cranepilot.data import load_dataset, sha256
from cranepilot.rollout import command_windows, HORIZON_INDEX, HORIZONS_S
from scripts.combined_candidate import CombinedCandidate
from scripts.combined_stress_simulator import simulate_stress
from scripts.combined_history_baseline import adaptive_free_forecast


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/runs/combined_stress_20261107'))
    args = parser.parse_args(); out = args.output
    config_path = Path('configs/combined_stress_20261107.json')
    cfg = json.loads(config_path.read_text())
    files = [config_path, Path('docs/COMBINED_STRESS_DESIGN_20261107.md'), Path(__file__).relative_to(Path.cwd()),
             *sorted(Path('scripts').glob('combined_*.py')), *sorted(Path('tests').glob('test_combined_*.py'))]
    for path in files:
        if subprocess.check_output(['git', 'show', f'HEAD:{path.as_posix()}']) != path.read_bytes():
            raise ValueError(f'Commit before evaluating: {path}')
    if out.exists(): raise FileExistsError('Preserve prior results')
    files += [*Path('src/cranepilot').glob('*.py'), *Path('scripts').glob('*.py'), Path('data/processed/dataset.npz'),
              Path('data/processed/manifest.json'), Path(cfg['source_run']) / 'provenance.json', Path(cfg['broad_run']) / 'provenance.json']
    hashes = {str(p): sha256(p) for p in sorted(set(files))}
    out.mkdir(parents=True)
    prov = {'status': 'running', 'config': cfg, 'guarded_hashes': hashes, 'environment': environment_record(),
            'started_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'training_performed': False,
            'real_evaluation_retrospective': True, 'candidate_frozen_before_new_simulation': True}
    write(out / 'provenance.json', prov)
    try:
        evaluate(cfg, out, prov)
    except Exception as error:
        prov.update(status='failed', failure=f'{type(error).__name__}: {error}')
        write(out / 'provenance.json', prov)
        raise


def evaluate(cfg, out, prov):
    torch.set_num_threads(4)
    old = json.loads((Path(cfg['source_run']) / 'provenance.json').read_text())
    data = load_dataset(Path('data/processed'))
    real_rows, sim_rows, gate_rows, checks = [], [], [], []
    for fold in cfg['folds']:
        predictor = CombinedCandidate(cfg, fold); cal = predictor.calibration
        directory = out / fold; directory.mkdir()
        split = old['config']['folds'][fold]
        train_lengths = [float(data[s].signals['L'][0]) for s in split['train']]
        for group, sids in [('validation', split['validation']), ('evaluation', split['evaluation']), ('load_stress', split['load_stress'])]:
            for sid in sids:
                s = data[sid]; k = np.arange(40, len(s.time) - 150, cfg['evaluation_stride'])
                raw = command_windows(s, cal, k)
                pred, seeds, gate = predictor.from_raw(raw)
                # Parent replay detects changed physical convention or model assembly.
                previous = Path(cfg['broad_run']) / fold / f'predictions_s{sid}.npz'
                with np.load(previous) as retained:
                    np.testing.assert_array_equal(k, retained['k'])
                    for name in ('incumbent', 'soft_symmetry'):
                        reference = retained[name]
                        if reference.ndim == 3: reference = reference.mean(0)
                        delta = float(np.abs(np.rad2deg(pred[name] - reference)).max())
                        if delta > 5e-5: raise AssertionError(('parent_replay', fold, sid, name, delta))
                        checks.append({'check': 'parent_replay', 'fold': fold, 'sid': sid, 'model': name, 'max_deg': delta})
                if group == 'validation':
                    indices = np.array([0, len(k)//2, len(k)-1]); origin = k[indices]; past = origin[:, None] - np.arange(41)
                    packet = {'theta': s.signals['theta'][past], 'x': s.signals['x'][past], 'reference': s.signals['xr'][past],
                              'length_m': s.signals['L'][origin], 'future_reference': s.signals['xr'][origin[:,None] + np.arange(30)*5]}
                    actual, _, _ = predictor.from_packet(packet)
                    delta = max(float(np.abs(np.rad2deg(actual[n]-y[indices])).max()) for n,y in pred.items())
                    if delta > 5e-5: raise AssertionError(('history_packet',delta))
                    small = {key: value[indices].copy() for key,value in raw.items()}
                    clean_subset, _, _ = predictor.from_raw(small)
                    small['target_q'][:] = 12345
                    small['target_x'][:] = -12345
                    poisoned, _, _ = predictor.from_raw(small)
                    for name in clean_subset: np.testing.assert_array_equal(clean_subset[name], poisoned[name])
                    checks.append({'check':'history_only_and_target_invariance','fold':fold,'max_deg':delta})
                active = np.abs(raw['initial_velocity']) >= cfg['active_threshold']
                length = float(s.signals['L'][0]); extra = length < min(train_lengths) or length > max(train_lengths)
                for name,y in pred.items():
                    error = np.abs(np.rad2deg(y[:,HORIZON_INDEX] - raw['target_q'][:,HORIZON_INDEX]))
                    for regime,mask in [('all',np.ones(len(y),bool)),('active',active),('quiet',~active)]:
                        if not mask.any(): continue
                        for j,h in enumerate(HORIZONS_S):
                            real_rows.append({'fold':fold,'group':group,'scenario':sid,'extrapolation':extra,'model':name,
                                              'regime':regime,'horizon_s':h,'mae_deg':float(error[mask,j].mean())})
                np.savez_compressed(directory / f'predictions_s{sid}.npz', k=k,target=raw['target_q'],gate=gate,**pred,
                                    **{n+'__seeds':v for n,v in seeds.items()})
                print('REAL',fold,sid,flush=True)
        meta,theta,target = simulate_stress(cfg['simulation'],cal)
        finer_meta,finer_theta,finer_target = simulate_stress(cfg['simulation'],cal,dt=.0005)
        delta = float(np.abs(np.rad2deg(target-finer_target)).max())
        if delta > 5e-5: raise AssertionError(('integrator',delta))
        checks.append({'check':'integrator_half_step','fold':fold,'max_deg':delta})
        packet = {'theta':theta,'x':np.zeros_like(theta),'reference':np.zeros_like(theta),
                  'length_m':meta.length_m.to_numpy(),'future_reference':np.zeros((len(theta),30))}
        pred,seeds,gate = predictor.from_packet(packet,autonomous=True)
        adaptive,metadata = adaptive_free_forecast(theta,cal.omega(meta.length_m.to_numpy()))
        pred['adaptive_physics'] = adaptive
        # Unannounced future forcing must not leak into any forecast or past noise.
        nominal = np.flatnonzero(meta.condition.to_numpy()=='nominal'); hidden = np.flatnonzero(meta.condition.to_numpy()=='hidden_impulse')
        np.testing.assert_array_equal(theta[nominal],theta[hidden])
        for name,y in pred.items(): np.testing.assert_array_equal(y[nominal],y[hidden])
        checks.append({'check':'identical_past_hidden_future','fold':fold,'pairs':len(nominal),'max_deg':0.0})
        for name,y in pred.items():
            error = np.abs(np.rad2deg(y[:,HORIZON_INDEX]-target[:,HORIZON_INDEX]))
            for j,h in enumerate(HORIZONS_S):
                local = meta.copy(); local['fold']=fold; local['model']=name; local['horizon_s']=h; local['absolute_error_deg']=error[:,j]
                sim_rows.append(local)
        for (condition,mode),positions in meta.groupby(['condition','measurement']).groups.items():
            g = gate[np.asarray(positions)]
            gate_rows.append({'fold':fold,'condition':condition,'measurement':mode,'mean_network_weight':float(g.mean()),
                              'p10':float(np.quantile(g,.1)),'p90':float(np.quantile(g,.9))})
        meta.to_csv(directory/'simulation_cases.csv',index=False)
        pd.DataFrame(metadata).to_csv(directory/'adaptive_choices.csv',index=False)
        np.savez_compressed(directory/'simulation_predictions.npz',theta=theta,target=target,gate=gate,**pred,
                            **{n+'__seeds':v for n,v in seeds.items()})
        print('SIMULATION',fold,len(meta),'cases',flush=True)
    pd.DataFrame(real_rows).to_csv(out/'real_metrics.csv',index=False)
    pd.concat(sim_rows,ignore_index=True).to_csv(out/'simulation_metrics.csv',index=False)
    pd.DataFrame(gate_rows).to_csv(out/'gate_statistics.csv',index=False)
    write(out/'verification.json',{'passed':True,'tolerance_deg':5e-5,'checks':checks})
    prov.update(status='complete',finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                result_hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'})
    write(out/'provenance.json',prov)
    print('COMPLETE',out,flush=True)


if __name__ == '__main__': main()
