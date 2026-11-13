"""Verify original scientific artifacts and stored metrics without training."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'chaos-forecasting-ai/physical_forecasting'))
from evaluation import summarize_file


def verify_checksums(root):
    records=(root/'provenance/SHA256SUMS').read_text().splitlines()
    for record in records:
        expected,relative=record.split('  ',1)
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise AssertionError(f'Invalid or missing artifact: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise AssertionError(f'Checksum mismatch: {relative}')
    return len(records)


def verify_laboratory(root):
    base=root/'chaos-forecasting-ai/results/long-horizon-2026-11-05'
    cases=cutoffs=0
    for system in ['lorenz','rossler','van-der-pol']:
        directory=base/system
        config=json.loads((directory/'config.json').read_text())
        metrics=json.loads((directory/'metrics.json').read_text())
        d=config['physical_size']
        with np.load(directory/'data.npz',allow_pickle=False) as data:
            truth=data['truth'][:,:,:d]
            scales=data['train'][:,:,:d].std(axis=(0,1))
            assert data['train'].shape[0]==32
            assert data['validation'].shape[0]==4
        for family,label in [('polynomial-dynamics','Polynomial dynamics'),('transformer','Transformer'),('gru','GRU'),('mlp','MLP')]:
            with np.load(directory/(family+'.npz'),allow_pickle=False) as data:
                members=data['members'].copy()
                assert data['mean'].shape[:2]==truth.shape[:2]
            for member in members:
                normalized=member[:,:,:d]/scales
                bad=~np.isfinite(normalized).all(axis=2)|(np.abs(normalized).max(axis=2)>1e6)
                member[np.maximum.accumulate(bad,axis=1)]=np.nan
            error=members.mean(axis=0)[:,:,:d]-truth
            for time,score in metrics[label]['point_by_cutoff'].items():
                n=min(round(float(time)/config['dt']),truth.shape[1])
                values=error[:,:n]
                if np.isfinite(values).all():
                    np.testing.assert_allclose(np.sqrt(np.mean(values**2,axis=(0,1))),score['rmse_by_state'],rtol=1e-10,atol=1e-10)
                    np.testing.assert_allclose(np.sqrt(np.mean((values/scales)**2)),score['normalized_rmse_all_states'],rtol=1e-10,atol=1e-10)
                else:
                    assert score['rmse_by_state'] is None
                    assert score['normalized_rmse_all_states'] is None
                cutoffs+=1
            cases+=len(truth)
    return cases,cutoffs


def verify_physical(root):
    base=root/'chaos-forecasting-ai/results/physical-forecasting'
    cases=nonfinite=0
    for slug in ['pendulum','waterwheel']:
        baseline=base/'run-2026-11-05'/slug
        config=json.loads((baseline/'summary.json').read_text())
        for family in ['mlp','polynomial-dynamics','transformer','gru']:
            directory=baseline if family=='mlp' else base/'comparison-2026-11-12'/slug/family
            actual=summarize_file(directory/'forecast.npz',slug,config['dt'],config['trainingScales'])
            expected=json.loads((directory/'scientific-metrics.json').read_text())
            assert actual==expected,(slug,family)
            cases+=len(actual['runs'])
            nonfinite+=sum(run['nonfiniteValues'] for run in actual['runs'])
    return cases,nonfinite


def verify(root=ROOT):
    root=Path(root).resolve()
    count=verify_checksums(root)
    lab,cutoffs=verify_laboratory(root)
    physical,nonfinite=verify_physical(root)
    report={'checksummedFiles':count,'originalLaboratoryForecastOrigins':lab,
            'laboratoryCutoffSummaries':cutoffs,'physicalForecastRuns':physical,
            'nonfiniteRawPhysicalValuesRetained':nonfinite}
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    verify(parser.parse_args().root)
