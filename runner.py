from typing import Dict, List, Optional

import bytecode
import emit_IR
import scheme
from bytecode import BasicBlock, Binop, Function, Inst, Var
from emit_IR import FunctionEmitter
from scheme import Nil, SExp, SFunction, SSym, Value


def add_intrinsics(env: Dict[SSym, Value]) -> None:
    """Add intrinsics to the environment."""
    def inst_function(
            name: SSym, params: List[Var],
            return_to: Optional[Var], *insts: Inst,
            ) -> SFunction:
        """Create a function out of the instructions in insts."""
        begin = BasicBlock('bb0')
        for inst in insts:
            begin.add_inst(inst)
        if return_to is not None:
            begin.add_inst(bytecode.ReturnInst(return_to))
        code = Function(params, begin)
        param_syms = [SSym(p.name) for p in params]
        return SFunction(name, param_syms, Nil, code, False)

    def binop(name: SSym, op: Binop) -> SFunction:
        return inst_function(
            name, [Var('a'), Var('b')], result,
            bytecode.BinopInst(result, op, Var('a'), Var('b')))

    result = Var('result')
    env[SSym('inst/typeof')] = inst_function(
        SSym('inst/typeof'), [Var('x')], result,
        bytecode.TypeofInst(result, Var('x')))
    env[SSym('inst/trap')] = inst_function(
        SSym('inst/trap'), [], None,
        bytecode.TrapInst("(trap)"))
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


def add_builtins(env: Dict[SSym, Value]) -> None:
    """Add builtins to the environment."""
    code = scheme.parse("""
    (define (trap) (inst/trap))
    (define (assert b) (if b 0 (trap)))
    (define (typeof x) (inst/typeof x))
    (define (not b) (if b false true))

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
    emitter = FunctionEmitter(env)
    for definition in code:
        emitter.visit(definition)


def add_prelude(env: Dict[SSym, Value]) -> None:
    """Add prelude functions to the environment."""
    code = scheme.parse("""
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
    emitter = FunctionEmitter(env)
    for definition in code:
        emitter.visit(definition)


name_counter = 0


def run_code(env: Dict[SSym, Value], code: SExp) -> Value:
    """Run a piece of code in an environment, returning its result."""
    emitter = FunctionEmitter(env)
    if isinstance(code, scheme.SFunction):
        emitter.visit(code)
        return env[code.name]
    else:
        name = SSym(f'__eval_expr{name_counter}')
        code = scheme.SFunction(
            name, [], scheme.to_slist([code]), is_lambda=True)
        emitter.visit(code)
        function = env[name]
        assert isinstance(function, scheme.SFunction)
        assert function.code is not None
        eval_env = bytecode.EvalEnv({}, env)
        gen = bytecode.ResultGenerator(function.code.run(eval_env))
        gen.run()
        assert gen.value is not None
        return gen.value


def run(env: Dict[SSym, Value], text: str) -> Value:
    """
    Run a piece of code in an environment, returning its result.

    >>> env = {}
    >>> add_intrinsics(env)
    >>> add_builtins(env)
    >>> add_prelude(env)
    >>> run(env, '(+ 1 1)')
    SNum(value=2)
    >>> run(env, '(> (vector-length (cons 1 [])) 3)')
    SBool(value=False)
    """
    code = scheme.parse(text)
    result: Value = scheme.SVect([])
    for part in code:
        result = run_code(env, part)
    return result
