"""Stable command dispatch; command-specific flags are passed through unchanged."""
import argparse
import runpy
import sys

COMMANDS = {
    'prepare': 'urop.data.transcripts',
    'features': 'urop.features.pipeline',
    'train': 'urop.training.runner',
    'test-features': 'urop.evaluation.test_features',
    'evaluate': 'urop.evaluation.runner',
    'report': 'urop.training.report',
    'test-report': 'urop.evaluation.report',
    'legacy': 'urop.legacy',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        parser.parse_args()
        return
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [COMMANDS[args.command], *sys.argv[2:]]
    runpy.run_module(COMMANDS[args.command], run_name='__main__')


if __name__ == '__main__':
    main()
