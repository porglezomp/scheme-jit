#! /usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

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
    with Output(args) as out_f:
        args.out_file = out_f.file
        if args.machine_readable:
            report_stats_json(args, env)
        else:
            report_stats(args, env)


class Output:
    def __init__(self, args: argparse.Namespace) -> None:
        self.name = args.output[0]
        self.file = sys.stdout

    def __enter__(self) -> Output:
        if self.name != '-':
            self.file = open(self.name, 'w')
        return self

    def __exit__(self, *args: Any) -> None:
        if self.name != '-':
            self.file.close()


def report_stats_json(args: argparse.Namespace, env: bytecode.EvalEnv) -> None:
    raise NotImplementedError


def report_stats(args: argparse.Namespace, env: bytecode.EvalEnv) -> None:
    if args.function_stats:
        print('-----', file=args.out_file)
        for name, defn in env._global_env.items():
            assert isinstance(defn, sexp.SFunction)
            count = env.stats.function_count[id(defn.code)]
            params = ''.join(' ' + p.name for p in defn.params)
            print(f"{count:>8} ({defn.name}{params})", file=args.out_file)
            for types, spec in defn.specializations.items():
                count = env.stats.function_count[id(spec)]
                type_names = ', '.join(str(t) for t in types)
                print(f"{count:>8} ({defn.name}{params}) ({type_names})",
                      file=args.out_file)
    if args.all_stats:
        print('-----', file=args.out_file)
        for name, defn in env._global_env.items():
            assert isinstance(defn, sexp.SFunction)
            assert defn.code is not None
            count = env.stats.function_count[id(defn.code)]
            if count:
                print(defn.code.format_stats(name, None, env.stats),
                      file=args.out_file)
                print(file=args.out_file)
                for types, spec in defn.specializations.items():
                    print(spec.format_stats(name, types, env.stats),
                          file=args.out_file)
                    print(file=args.out_file)
    if args.stats:
        print('-----', file=args.out_file)
        total = 0
        for inst, count in env.stats.inst_type_count.items():
            total += count
            print(f"{inst.__name__:>10}: {count}", file=args.out_file)
        print(f"{'Total':>10}: {total}", file=args.out_file)
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
        '-S', '--print-specializations', action='store_true',
        help="log when specializations are created")
    parser.add_argument(
        '-O', '--print-optimizations', action='store_true',
        help="log when optimizations are performed")
    parser.add_argument(
        '-o', '--output', nargs=1, default='-',
        help="output the stats to a file")
    parser.add_argument(
        '-m', '--machine-readable', action='store_true',
        help="output stats as json")

    return parser.parse_args()


if __name__ == '__main__':
    main()
