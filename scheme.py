#! /usr/bin/env python3

import argparse
import sys
from typing import Dict

import bytecode
import runner
import sexp


def main() -> None:
    args = parse_args()
    if args.filename == '-':
        prog_text = sys.stdin.read()
    else:
        with open(args.filename) as f:
            prog_text = f.read()

    env = bytecode.EvalEnv()
    runner.add_intrinsics(env)
    runner.add_builtins(env)
    runner.add_prelude(env)
    print(runner.run(env, prog_text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')

    return parser.parse_args()


if __name__ == '__main__':
    main()
