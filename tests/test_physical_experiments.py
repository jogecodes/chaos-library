"""Physical experiments run without any presentation application or exports."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

CODE=Path(__file__).resolve().parents[1]/'chaos-forecasting-ai/physical_forecasting'
sys.path.insert(0,str(CODE))


def states():
    t=np.arange(220)*.01
    q=np.column_stack([np.sin(t),np.cos(t),np.cos(t),-np.sin(t)])
    return np.stack([q,q*.8])


def test_mlp_fit_saves_only_numerical_experiment_artifacts(tmp_path):
    torch=pytest.importorskip('torch');torch.set_num_threads(2)
    training=importlib.import_module('train_physical')
    q=states()
    forecast,scales=training.fit('pendulum',q,q[:,23:],.01,1,'cpu',tmp_path)
    predicted=forecast(q[:,23],5)
    assert predicted.shape==(2,6,4)
    assert np.isfinite(predicted).all()
    assert (scales>0).all()
    assert {p.name for p in tmp_path.iterdir()}=={'model.pt','training.json'}


@pytest.mark.parametrize('family',['transformer','gru'])
def test_sequence_fit_runs_independently(family,tmp_path):
    torch=pytest.importorskip('torch');torch.set_num_threads(2)
    training=importlib.import_module('train_comparison')
    q=states();scales=q.std(axis=(0,1))
    args=SimpleNamespace(updates=1,device='cpu')
    model,info=training.fit_neural(family,'pendulum',q,q[:,:24],q[:,23:],scales,.01,args,tmp_path)
    prediction=training.neural_rollout(model,q[:,:24],5)
    assert prediction.shape==(2,6,4)
    assert np.isfinite(prediction).all()
    assert info['selectedUpdate']==1
    assert (tmp_path/'model.pt').is_file()


def test_nonfinite_physical_outputs_are_retained_and_not_given_full_scores():
    from evaluation import summarize_run
    truth=np.zeros((201,4));predicted=truth.copy();predicted[190:]=np.nan
    report=summarize_run(truth,predicted,'pendulum',.01,np.ones(4))
    assert report['samples']==201
    assert report['firstNonfiniteTime']==pytest.approx(1.9)
    assert report['earlyMotionRmse']==0
    assert report['fullMotionRmse'] is None
    assert report['nonfiniteValues']==44

def test_comparison_runner_writes_complete_scientific_artifacts(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    import json
    training=importlib.import_module('train_comparison')
    baseline=importlib.import_module('train_physical')
    monkeypatch.setattr(training,'FAMILIES',('polynomial-dynamics',))
    source=tmp_path/'input'/'pendulum';source.mkdir(parents=True)
    initials=np.array([[.2,.1,0.,0.],[.3,-.1,0.,0.]])
    dt=.01;warmup=.24;duration=.05;origin=round(warmup/dt)
    paths=lambda length:np.stack([baseline.simulate(baseline.pendulum_rhs,q,length,dt) for q in initials])
    train=paths(2.2)
    validation=paths(warmup+3)[:,origin:]
    truth=paths(warmup+duration)[:,origin:]
    config={'dt':dt,'warmup':warmup,'duration':duration,'trainingScales':train.std(axis=(0,1)).tolist()}
    (source/'summary.json').write_text(json.dumps(config))
    (source/'initials.json').write_text(json.dumps({'validation':initials.tolist(),'test':initials.tolist()}))
    np.savez(source/'data.npz',train=train,validation=validation,truth=truth)
    output=tmp_path/'output'
    training.run(SimpleNamespace(system='pendulum',source=source.parent,artifacts=output,updates=1,device='cpu'))
    model=output/'pendulum/polynomial-dynamics'
    with np.load(model/'forecast.npz') as values:
        np.testing.assert_array_equal(values['truth'],truth)
        assert values['prediction'].shape==truth.shape
    assert (model/'model.npz').is_file()
    assert (model/'scientific-metrics.json').is_file()
    assert {p.name for p in (output/'pendulum/source').iterdir()}=={'train_comparison.py','train_physical.py','physical_io.py','evaluation.py'}
    assert all(p.suffix in {'.npz','.json','.py','.pt'} for p in output.rglob('*') if p.is_file())
