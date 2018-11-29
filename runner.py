from typing import Dict, List, Optional

import bytecode
import emit_IR
import sexp
from bytecode import BasicBlock, Binop, EvalEnv, Function, Inst, Var
from emit_IR import FunctionEmitter
from find_tail_calls import TailCallFinder
from sexp import Nil, SExp, SFunction, SSym, Value


def add_intrinsics(eval_env: EvalEnv) -> None:
    """Add intrinsics to the environment."""
    def inst_function(
            name: SSym, params: List[Var],
            return_val: Optional[bytecode.Parameter], *insts: Inst,
            ) -> SFunction:
        """Create a function out of the instructions in insts."""
        begin = BasicBlock('bb0')
        for inst in insts:
            begin.add_inst(inst)
        if return_val is not None:
            begin.add_inst(bytecode.ReturnInst(return_val))
        code = Function(params, begin)
        param_syms = [SSym(p.name) for p in params]
        return SFunction(name, param_syms, Nil, code, False)

    def binop(name: SSym, op: Binop) -> SFunction:
        return inst_function(
            name, [Var('a'), Var('b')], result,
            bytecode.BinopInst(result, op, Var('a'), Var('b')))

    result = Var('result')
    env = eval_env._global_env
    env[SSym('inst/typeof')] = inst_function(
        SSym('inst/typeof'), [Var('x')], result,
        bytecode.TypeofInst(result, Var('x')))
    env[SSym('inst/trap')] = inst_function(
        SSym('inst/trap'), [], None,
        bytecode.TrapInst("(trap)"))
    env[SSym('inst/trace')] = inst_function(
        SSym('inst/trace'), [Var('x')], bytecode.NumLit(sexp.SNum(0)),
        bytecode.TraceInst(Var('x')))
    env[SSym('inst/breakpoint')] = inst_function(
        SSym('inst/breakpoint'), [], bytecode.NumLit(sexp.SNum(0)),
        bytecode.BreakpointInst())
    # Memory operations
    env[SSym('inst/alloc')] = inst_function(
        SSym('inst/alloc'), [Var('n')], result,
        bytecode.AllocInst(result, Var('n')))
    env[SSym('inst/load')] = inst_function(
        SSym('inst/load'), [Var('v'), Var('n')], result,
        bytecode.LoadInst(result, Var('v'), Var('n')))
    env[SSym('inst/store')] = inst_function(
        SSym('inst/store'), [Var('v'), Var('n'), Var('x')], Var('v'),
        bytecode.StoreInst(Var('v'), Var('n'), Var('x')))
    env[SSym('inst/length')] = inst_function(
        SSym('inst/length'), [Var('v')], result,
        bytecode.LengthInst(result, Var('v')))
    # Binary operators
    env[SSym('inst/+')] = binop(SSym('inst/+'), Binop.ADD)
    env[SSym('inst/-')] = binop(SSym('inst/-'), Binop.SUB)
    env[SSym('inst/*')] = binop(SSym('inst/*'), Binop.MUL)
    env[SSym('inst//')] = binop(SSym('inst//'), Binop.DIV)
    env[SSym('inst/%')] = binop(SSym('inst/%'), Binop.MOD)
    env[SSym('inst/number=')] = binop(SSym('inst/number='), Binop.NUM_EQ)
    env[SSym('inst/symbol=')] = binop(SSym('inst/symbol='), Binop.SYM_EQ)
    env[SSym('inst/pointer=')] = binop(SSym('inst/pointer='), Binop.PTR_EQ)
    env[SSym('inst/number<')] = binop(SSym('inst/number<'), Binop.NUM_LT)


def add_builtins(env: EvalEnv) -> None:
    """Add builtins to the environment."""
    code = sexp.parse("""
    (define (trap) (inst/trap))
    (define (trace x) (inst/trace x))
    (define (breakpoint) (inst/breakpoint))
    (define (assert b) (if b 0 (trap)))
    (define (typeof x) (inst/typeof x))
    (define (not b)
      (assert (bool? b))
      (if b false true))

    (define (number? x) (symbol= (typeof x) 'number))
    (define (symbol? x) (symbol= (typeof x) 'symbol))
    (define (vector? x) (symbol= (typeof x) 'vector))
    (define (function? x) (symbol= (typeof x) 'function))
    (define (bool? x) (symbol= (typeof x) 'bool))
    (define (pair? x)
      (if (vector? x)
        (number= (vector-length x) 2)
        false))
    (define (nil? x)
      (if (vector? x)
        (number= (vector-length x) 0)
        false))

    (define (+ a b) (assert (number? a)) (assert (number? b)) (inst/+ a b))
    (define (- a b) (assert (number? a)) (assert (number? b)) (inst/- a b))
    (define (* a b) (assert (number? a)) (assert (number? b)) (inst/* a b))
    (define (/ a b)
      (assert (number? a))
      (assert (number? b))
      (assert (not (number= b 0)))
      (inst// a b))
    (define (% a b)
      (assert (number? a))
      (assert (number? b))
      (assert (not (number= b 0)))
      (inst/% a b))

    (define (pointer= a b) (inst/pointer= a b))
    (define (symbol= a b)
      ; Explicitly use inst/symbol= instead of symbol? because
      ; symbol? uses symbol=, so we need to avoid infinite recursion.
      (assert (inst/symbol= (typeof a) 'symbol))
      (assert (inst/symbol= (typeof b) 'symbol))
      (inst/symbol= a b))
    (define (number= a b)
      (assert (number? a))
      (assert (number? b))
      (inst/number= a b))

    (define (number< a b)
      (assert (number? a))
      (assert (number? b))
      (inst/number< a b))

    (define (vector-length v)
      (assert (vector? v))
      (inst/length v))

    (define (vector-index v n)
      (assert (vector? v))
      (assert (number? n))
      (assert (number< -1 n))
      (assert (number< n (vector-length v)))
      (inst/load v n))

    (define (vector-set! v n x)
      (assert (vector? v))
      (assert (number? n))
      (assert (number< -1 n))
      (assert (number< n (vector-length v)))
      (inst/store v n x))

    ;; The loop body for vector-make.
    ;; Fills the allocated vector. Not to be used on its own.
    (define (vector-make/recur len idx v x)
      (vector-set! v idx x)
      (if (number= len (+ idx 1))
        v
        (vector-make/recur len (+ idx 1) v x)))
    (define (vector-make n x)
      (assert (number? n))
      (assert (number< -1 n))
      (if (number= n 0)
        (inst/alloc 0)
        (vector-make/recur n 0 (inst/alloc n) x)))
    """)
    emitter = FunctionEmitter(env._global_env)
    for definition in code:
        emitter.visit(definition)


def add_prelude(env: EvalEnv) -> None:
    """Add prelude functions to the environment."""
    code = sexp.parse("""
    ;; The loop body for vector=, not to be used on its own
    (define (vector=/recur x y n end)
      (if (= n end)
        true
        (if (= (vector-index x n) (vector-index y n))
          (vector=/recur x y (+ n 1) end)
          false)))
    (define (vector= x y)
      (assert (vector? x))
      (assert (vector? y))
      (if (pointer= x y)
        true
        (if (= (vector-length x) (vector-length y))
            (vector=/recur x y 0 (vector-length x))
            false)))

    (define (= x y)
      (if (not (symbol= (typeof x) (typeof y)))
        false
        (if (symbol? x)
          (symbol= x y)
          (if (number? x)
            (number= x y)
            (if (vector? x)
              (vector= x y)
              (pointer= x y))))))

    (define (< x y) (number< x y))

    (define (!= x y) (not (= x y)))
    (define (> x y) (< y x))
    (define (<= x y) (if (= x y) true (< x y)))
    (define (>= x y) (if (= x y) true (> x y)))

    (define (cons x l) [x l])
    (define (car l) (vector-index l 0))
    (define (cdr l) (vector-index l 1))
    """)
    emitter = FunctionEmitter(env._global_env)
    for definition in code:
        emitter.visit(definition)


eval_names = emit_IR.name_generator('__eval_expr')


def run_code(env: EvalEnv, code: SExp,
             optimize_tail_calls: bool = False) -> Value:
    """Run a piece of code in an environment, returning its result."""
    emitter = FunctionEmitter(env._global_env)
    if isinstance(code, sexp.SFunction):
        tail_calls = None
        if optimize_tail_calls:
            tail_call_finder = TailCallFinder()
            tail_call_finder.visit(code)
            tail_calls = tail_call_finder.tail_calls

        emitter = FunctionEmitter(env._global_env, tail_calls=tail_calls)
        emitter.visit(code)
        return env._global_env[code.name]
    else:
        name = SSym(f'{next(eval_names)}')
        code = sexp.SFunction(
            name, [], sexp.to_slist([code]), is_lambda=True)
        emitter = FunctionEmitter(env._global_env)
        emitter.visit(code)
        function = env._global_env[name]
        assert isinstance(function, sexp.SFunction)
        assert function.code is not None
        gen = bytecode.ResultGenerator(function.code.run(env))
        gen.run()
        assert gen.value is not None
        return gen.value


def run(env: EvalEnv, text: str, optimize_tail_calls: bool = False) -> Value:
    """
    Run a piece of code in an environment, returning its result.

    >>> env = EvalEnv()
    >>> add_intrinsics(env)
    >>> add_builtins(env)
    >>> add_prelude(env)
    >>> run(env, '(+ 1 1)')
    SNum(value=2)
    >>> run(env, '(> (vector-length (cons 1 [])) 3)')
    SBool(value=False)
    """
    code = sexp.parse(text)
    result: Value = sexp.SVect([])
    for part in code:
        result = run_code(env, part, optimize_tail_calls=optimize_tail_calls)
    return result
