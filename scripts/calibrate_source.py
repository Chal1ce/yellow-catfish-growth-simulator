"""Estimate two source shifts from one completed control."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aquabiosim.model import compile_engine, load_inputs, load_model, prepared, simulate, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--control-id', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/yellow_catfish.json')
    args = parser.parse_args()
    rows = load_inputs(args.inputs)
    controls = [r for r in rows if r['treatment_id'] == args.control_id]
    if len(controls) != 1:
        raise ValueError('control-id must identify exactly one treatment')
    control = controls[0]
    rows = [r for r in rows if r['study_id'] == control['study_id']]
    index = next(i for i, r in enumerate(rows) if r is control)
    target = val(control, 'wf_g') or val(control, 'w0_g') * val(control, 'observed_growth_ratio')
    if target <= 0 or not np.isfinite(target):
        raise ValueError('The completed control requires wf_g or observed_growth_ratio')
    channel, observed = None, None
    for name in ['observed_feed_g', 'feeding_rate_pct_bw_day', 'fcr_g_g']:
        if control.get(name, '') != '':
            channel, observed = name, float(control[name])
            if observed <= 0 or not np.isfinite(observed):
                raise ValueError('The control intake/FR/FCR value must be positive and finite')
            break
    penalty = 0. if channel else .25
    model, core = load_model(args.model)
    data = prepared(rows, model['process_heads'], 'nutrition_compact')
    lib = compile_engine(args.output.parent / 'runtime')

    def residual(shifts):
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                data.update(capacity_log_offset=float(shifts[0]), eta_offset=float(shifts[1]))
                result = simulate(core, data, lib)
                weight, intake = result['weight_g'][index], result['feed_g'][index]
                values = [np.log(weight / target)]
                if channel == 'observed_feed_g':
                    values.append(2 * np.log(intake / observed))
                elif channel == 'feeding_rate_pct_bw_day':
                    predicted = 100 * intake / (val(control, 'days') * (val(control, 'w0_g') + weight) / 2)
                    values.append(.5 * np.log(max(predicted, 1e-12) / observed))
                elif channel == 'fcr_g_g':
                    predicted = intake / max(weight - val(control, 'w0_g'), 1e-7)
                    values.append(.5 * np.log(max(predicted, 1e-12) / observed))
                return np.r_[values, np.sqrt(penalty) * shifts]
        except (ValueError, FloatingPointError):
            return np.full(3 + int(channel is not None), 1e3)

    fits = [least_squares(residual, start, bounds=([-4., -4.], [4., 4.]),
                          max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            for start in ([0., 0.], [-1., 0.], [1., -1.])]
    valid = [fit for fit in fits if np.max(abs(residual(fit.x))) < 100]
    if not valid:
        raise ValueError('No valid completed-control calibration')
    best = min(valid, key=lambda fit: fit.cost)
    result = dict(source_shifts={control['study_id']: dict(
        capacity_log_offset=float(best.x[0]), eta_offset=float(best.x[1]))},
        control_id=args.control_id, intake_channel=channel, penalty=penalty,
        status='converged' if best.success else 'iteration_limit_valid')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), status=result['status'])))


if __name__ == '__main__':
    main()
