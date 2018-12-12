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

    env = bytecode.EvalEnv(optimize_tail_calls=args.tail_calls,
                           jit=args.jit,
                           bytecode_jit=args.bytecode_jit,
                           print_specializations=args.print_specializations,
                           print_optimizations=args.print_optimizations)
    runner.add_intrinsics(env)
    runner.add_builtins(env)
    runner.add_prelude(env)
    print(runner.run(env, prog_text))

    if args.function_stats:
        print('-----')
        for name, defn in env._global_env.items():
            assert isinstance(defn, sexp.SFunction)
            count = env.stats.function_count[id(defn.code)]
            params = ''.join(' ' + p.name for p in defn.params)
            print(f"{count:>8} ({defn.name}{params})")
            for types, spec in defn.specializations.items():
                count = env.stats.function_count[id(spec)]
                type_names = ', '.join(str(t) for t in types)
                print(f"{count:>8} ({defn.name}{params}) ({type_names})")
    if args.all_stats:
        print('-----')
        for name, defn in env._global_env.items():
            assert isinstance(defn, sexp.SFunction)
            assert defn.code is not None
            count = env.stats.function_count[id(defn.code)]
            if count:
                print(defn.code.format_stats(name, None, env.stats))
                print()
                for types, spec in defn.specializations.items():
                    print(spec.format_stats(name, types, env.stats))
                    print()
    if args.stats:
        print('-----')
        total = 0
        for inst, count in env.stats.inst_type_count.items():
            total += count
            print(f"{inst.__name__:>10}: {count}")
        print(f"{'Total':>10}: {total}")
        # print('-----')
        # for inst, count in env.stats.inst_type_count.items():
        #     print(f"{inst.__name__:>10}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filename',
        help="the script to execute, or - for stdin")
    parser.add_argument(
        '-s', '--stats', action='store_true',
        help="print baseline stats")
    parser.add_argument(
        '-f', '--function-stats', action='store_true',
        help="print execution counts for every function")
    parser.add_argument(
        '-a', '--all-stats', action='store_true',
        help="print stats for every instruction of every function")
    parser.add_argument(
        '-j', '--jit', action='store_true',
        help="optimize at the AST level")
    parser.add_argument(
        '-b', '--bytecode-jit', action='store_true',
        help="optimize at the bytecode level")
    parser.add_argument(
        '-t', '--tail-calls', action='store_true',
        help="do tail call optimization")
    parser.add_argument(
        '-p', '--print-specializations', action='store_true',
        help="log when specializations are created")
    parser.add_argument(
        '-o', '--print-optimizations', action='store_true',
        help="log when optimizations are performed")

    return parser.parse_args()


if __name__ == '__main__':
    main()
