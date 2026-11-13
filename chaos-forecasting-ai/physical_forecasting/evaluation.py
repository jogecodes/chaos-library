"""Scientific metrics for complete physical forecasts, including invalid outputs."""
import hashlib
import numpy as np


def finite_rmse(values):
    if values.size == 0 or not np.isfinite(values).all():
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        result = float(np.sqrt(np.mean(values**2)))
    return result if np.isfinite(result) else None


def summarize_run(truth, prediction, slug, dt, scales):
    if truth.shape != prediction.shape:
        raise ValueError('Truth and forecast shapes differ')
    finite = np.isfinite(prediction).all(axis=1)
    invalid = np.flatnonzero(~finite)
    error = prediction-truth
    angles = 2 if slug == 'pendulum' else 1
    with np.errstate(invalid='ignore'):
        error[:, :angles] = np.arctan2(np.sin(error[:, :angles]), np.cos(error[:, :angles]))
        if slug == 'pendulum':
            def tip(q):
                return np.column_stack([np.sin(q[:, 0])+np.sin(q[:, 1]), -np.cos(q[:, 0])-np.cos(q[:, 1])])
            motion = np.linalg.norm(tip(prediction)-tip(truth), axis=1)
        else:
            motion = np.abs(error[:, 1])
    early = round(1.8/dt)
    negative = np.flatnonzero(np.any(prediction[:, 2:] < -1e-5, axis=1)) if slug == 'waterwheel' else np.array([], dtype=int)
    return {'samples':len(truth),'duration':(len(truth)-1)*dt,
            'allStatesFinite':bool(finite.all()),
            'firstNonfiniteTime':float(invalid[0]*dt) if len(invalid) else None,
            'nonfiniteValues':int(np.count_nonzero(~np.isfinite(prediction))),
            'firstNegativeBucketMassTime':float(negative[0]*dt) if len(negative) else None,
            'earlyMotionRmse':finite_rmse(motion[1:early+1]) if len(truth)>early else None,
            'fullMotionRmse':finite_rmse(motion[1:]),
            'earlyNrmse':finite_rmse(error[1:early+1]/scales) if len(truth)>early else None,
            'fullNrmse':finite_rmse(error[1:]/scales)}


def summarize_file(path, slug, dt, scales):
    with np.load(path, allow_pickle=False) as data:
        truth, prediction = data['truth'], data['prediction']
    return {'forecastSha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'runs':[{'index':i+1, **summarize_run(t,p,slug,dt,np.asarray(scales))}
                    for i,(t,p) in enumerate(zip(truth,prediction))]}
