"""Import experiment inputs and optional saved results from the curated data ZIP."""
import argparse
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'yellow_catfish_data_package'
TABLES = ('treatments.csv', 'process_targets.csv', 'secondary_targets.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--data-only', action='store_true',
                        help='Import inputs only; start fresh fits in a new checkout.')
    args = parser.parse_args()
    count = 0
    with zipfile.ZipFile(args.archive) as archive:
        for table in TABLES:
            archive.getinfo(f'{PACKAGE}/01_raw_extracted_data/{table}')
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if info.is_dir() or len(parts) < 3 or parts[0] != PACKAGE:
                continue
            if '..' in parts or '\\' in info.filename:
                raise ValueError(f'Invalid archive member: {info.filename}')
            section, tail = parts[1], Path(*parts[2:])
            if section == '01_raw_extracted_data' and str(tail) in TABLES:
                dest = ROOT / 'data/processed/multisource_v0.18' / tail
            elif section == '03_figure_source_data':
                dest = ROOT / 'data/figure_source_data' / tail
            elif section == '02_model_predictions' and not args.data_only:
                dest = ROOT / 'artifacts/reviewer-revision-v0.46' / tail
            else:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(info))
            count += 1
    print(f'Imported {count} files into {ROOT}')


if __name__ == '__main__':
    main()
