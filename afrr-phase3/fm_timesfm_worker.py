"""
fm_timesfm_worker.py — TimesFM 2.5 200M inference, run inside the existing
`timesfm` conda env (which has no pyarrow, so it never reads the parquet).

Called by foundation.py --model timesfm. Input: an .npz with `context`
(N, 2016) float64 and scalar `pred_len`. Output: an .npz with `quantiles`
(N, pred_len, 9) for levels 0.1..0.9, `seconds` (N,) wall-clock per origin
(call time / origins in the call), `call_seconds` per forecast call, `peak_rss_bytes`, `device`, `torch_version`, `timesfm_version`.

--batch origins per model.forecast call (per_core_batch_size = batch; the
library pads a short last batch with dummy series and drops their output).
The context is passed as float32 (the library casts to float32 anyway; a
float64 array fails on MPS because it is moved to the device before the cast).
ForecastConfig as in afrr-phase2/experiment_2a.py, except max_context = 2016
(the shared 3-week context) and max_horizon = 128 (pred_len rounded up to the
output patch, as compile() does itself).

Usage: <timesfm-env python> fm_timesfm_worker.py IN.npz OUT.npz --device cpu|mps --threads 8 --batch 1
"""
import argparse
import resource
import sys
import time

import numpy as np
import timesfm
import torch

CONTEXT = 2016


def load(device, batch):
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
    if device != 'cpu':
        # the library only picks cuda or cpu; move module and its device attribute
        model.model.device = torch.device(device)
        model.model.to(device)
    model.compile(timesfm.ForecastConfig(
        max_context=CONTEXT,
        max_horizon=128,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,      # aFRR prices can be negative
        fix_quantile_crossing=True,
        per_core_batch_size=batch,
    ))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp')
    ap.add_argument('out')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--batch', type=int, default=1, help='origins per forecast call')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = np.load(a.inp)
    ctx, pred_len = d['context'], int(d['pred_len'])
    assert ctx.shape[1] == CONTEXT and not np.isnan(ctx).any()
    model = load(a.device, a.batch)
    n = len(ctx)
    out = np.empty((n, pred_len, 9), np.float32)
    sec = np.empty(n)
    calls = []
    for b0 in range(0, n, a.batch):
        b1 = min(n, b0 + a.batch)
        t = time.perf_counter()
        _, q = model.forecast(horizon=pred_len, inputs=[x.astype(np.float32) for x in ctx[b0:b1]])
        dt = time.perf_counter() - t
        out[b0:b1] = q[:, :, 1:10]           # index 0 = mean, 1..9 = 0.1..0.9
        sec[b0:b1] = dt / (b1 - b0)
        calls.append(dt)
    np.savez(a.out, quantiles=out, seconds=sec, call_seconds=np.array(calls),
             peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             device=a.device, torch_version=torch.__version__,
             timesfm_version=getattr(timesfm, '__version__', 'unknown'))


if __name__ == '__main__':
    sys.exit(main())
