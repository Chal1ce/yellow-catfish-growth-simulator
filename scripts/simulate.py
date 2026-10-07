"""Run the frozen model on user-provided treatment inputs."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aquabiosim.model import (apply_source_shifts, compile_engine, load_inputs,
                             load_model, prepared, simulate, write_predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/yellow_catfish.json')
    parser.add_argument('--mode', choices=['zero_context', 'learned_context'], default='zero_context')
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    if args.calibration and args.mode != 'zero_context':
        parser.error('Use --calibration without --mode learned_context')
    rows = load_inputs(args.inputs)
    model, core = load_model(args.model)
    data = prepared(rows, model['process_heads'], 'nutrition_compact')
    if args.calibration:
        apply_source_shifts(data, json.loads(args.calibration.read_text())['source_shifts'])
    elif args.mode == 'learned_context':
        apply_source_shifts(data, model['source_shifts'])
    lib = compile_engine(args.output.parent / 'runtime')
    result = simulate(core, data, lib)
    write_predictions(args.output, rows, result, data)
    print(json.dumps(dict(predictions=len(rows), output=str(args.output))))


if __name__ == '__main__':
    main()
