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

    env = bytecode.EvalEnv(optimize_tail_calls=args.optimize_tail_calls,
                           naive_jit=args.naive_jit,
                           bytecode_jit=args.bytecode_jit,
                           print_specializations=args.print_specializations)
    runner.add_intrinsics(env)
    runner.add_builtins(env)
    runner.add_prelude(env)
    print(runner.run(env, prog_text))

    if args.stats:
        print('-----')
        for name, defn in env._global_env.items():
            if name.name.startswith('__eval_expr'):
                continue
            assert isinstance(defn, sexp.SFunction)
            assert defn.code is not None
            count = env.stats.function_count[id(defn.code)]
            if count:
                print(defn.code.format_stats(name, env.stats))
                print()
        print('-----')
        for inst, count in env.stats.inst_type_count.items():
            print(f"{inst.__name__:>10}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filename',
        help="the script to execute, or - for stdin")
    parser.add_argument(
        '-s', '--stats', action='store_true',
        help="print execution stats")

    parser.add_argument(
        '--optimize_tail_calls', '-t', action='store_true', default=False)

    parser.add_argument(
        '--naive_jit', '-n', action='store_true', default=False)

    parser.add_argument(
        '--bytecode_jit', '-b', action='store_true', default=False
    )

    parser.add_argument('--print_specializations', '-p',
                        action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    main()
