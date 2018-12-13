#! /usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from typing import Any, Dict, List

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

    env = bytecode.EvalEnv(
        optimize_tail_calls=args.tail_calls,
        jit=args.jit,
        bytecode_jit=args.bytecode_jit,
        print_specializations=args.print_specializations,
        print_optimizations=args.print_optimizations,
        inline_threshold=args.inline_count,
        specialization_threshold=args.specialize_count,
    )
    start = time.perf_counter()
    runner.add_intrinsics(env)
    runner.add_builtins(env)
    runner.add_prelude(env)
    startup = time.perf_counter()
    print(runner.run(env, prog_text))
    end = time.perf_counter()
    env.stats.startup_time = startup - start
    env.stats.program_time = end - startup
    with Output(args) as out_f:
        args.out_file = out_f.file
        if args.machine_readable:
            report_stats_json(args, env)
        else:
            report_stats(args, env)


class Output:
    def __init__(self, args: argparse.Namespace) -> None:
        self.name = args.output
        self.file = sys.stdout

    def __enter__(self) -> Output:
        if self.name != '-':
            self.file = open(self.name, 'w')
        return self

    def __exit__(self, *args: Any) -> None:
        if self.name != '-':
            self.file.close()


def report_stats_json(args: argparse.Namespace, env: bytecode.EvalEnv) -> None:
    result: Dict[str, Any] = {
        'config': {
            'tail_calls': env.optimize_tail_calls,
            'ast_jit': env.jit,
            'bytecode_jit': env.bytecode_jit,
            'inline_threshold': env.inline_threshold,
            'specialization_threshold': env.specialization_threshold,
        }
    }
    if args.function_stats:
        functions: Dict[str, Any] = {}
        result['functions'] = functions
        for name, defn in env._global_env.items():
            assert isinstance(defn, sexp.SFunction)
            count = env.stats.function_count[id(defn.code)]
            params = ''.join(' ' + p.name for p in defn.params)
            specializations: Dict[str, Any] = {}
            functions[defn.name.name] = {
                'params': [str(p) for p in defn.params],
                'count': count,
                'total': count + sum(
                    env.stats.function_count[id(spec)]
                    for spec in defn.specializations.values()),
                'specializations': specializations,
            }
            for types, spec in defn.specializations.items():
                count = env.stats.function_count[id(spec)]
                type_names = [str(t) for t in types]
                spec_name = f"({', '.join(type_names)})"
                specializations[spec_name] = {
                    'types': type_names,
                    'count': count,
                }
    if args.all_stats:
        functions = {}
        result['code'] = functions
        for name, defn in env._global_env.items():
            assert isinstance(defn, sexp.SFunction)
            assert defn.code is not None
            specializations = {}
            functions[defn.name.name] = {
                'code': defn.code.format_stats(name, None, env.stats),
                'specializations': specializations,
            }
            for types, spec in defn.specializations.items():
                type_names = [str(t) for t in types]
                spec_name = f"({', '.join(type_names)})"
                specializations[spec_name] = (
                    spec.format_stats(name, types, env.stats))
    if args.stats:
        inst_counts: Dict[str, int] = {}
        result['inst_counts'] = inst_counts
        total = 0
        for inst, count in env.stats.inst_type_count.items():
            total += count
            inst_counts[inst.__name__] = count
        inst_counts['Total'] = total
        result['time'] = {
            'startup': env.stats.startup_time,
            'program': env.stats.program_time,
        }
    json.dump(result, args.out_file, indent=2)


def report_stats(args: argparse.Namespace, env: bytecode.EvalEnv) -> None:
    with contextlib.redirect_stdout(args.out_file):
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
                    print(f"{count:>8}   ({defn.name}{params}) ({type_names})")
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
            print('-----')
            print(f"Startup time: {env.stats.startup_time:0.2f}")
            print(f"Program time: {env.stats.program_time:0.2f}")
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
        '-o', '--output', metavar='FILE',
        type=str, default='-',
        help="output the stats to a file")
    parser.add_argument(
        '-m', '--machine-readable', action='store_true',
        help="output stats as json")
    parser.add_argument(
        '-i', '--inline-count', metavar='COUNT',
        type=int, default=10,
        help="the threshold to stop inlining at (default 10)")
    parser.add_argument(
        '-c', '--specialize-count', metavar='COUNT',
        type=int, default=2,
        help="the number of executions before creating a specialization "
             "(default 2)")

    return parser.parse_args()


if __name__ == '__main__':
    main()
