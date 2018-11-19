#! /usr/bin/env python3

import argparse
from typing import Dict

import runner
import sexp


def main() -> None:
    args = parse_args()
    with open(args.filename) as f:
        env: Dict[sexp.SSym, sexp.Value] = {}
        runner.add_intrinsics(env)
        runner.add_builtins(env)
        runner.add_prelude(env)
        print(runner.run(env, f.read()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')

    return parser.parse_args()


if __name__ == '__main__':
    main()
