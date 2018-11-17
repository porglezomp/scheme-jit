from typing import Dict, List

import bytecode
import scheme
from bytecode import BasicBlock, Binop, Function, Inst, Var
from emit_IR import FunctionEmitter
from scheme import Nil, SFunction, SSym, Value


def inst_function(
        name: SSym, params: List[Var], *insts: Inst,
        should_return: bool = True) -> SFunction:
    """Create a function that's just one instruction."""
    begin = BasicBlock('bb0')
    for inst in insts:
        begin.add_inst(inst)
    if should_return:
        begin.add_inst(bytecode.ReturnInst(Var('result')))
    code = Function(params, begin)
    return SFunction(name, [SSym(p.name) for p in params], Nil, code, False)


def add_intrinsics(env: Dict[SSym, Value]) -> None:
    """Add intrinsics to the environment."""
    result = Var('result')
    env[SSym('inst/typeof')] = inst_function(
        SSym('inst/typeof'), [Var('x')],
        bytecode.TypeofInst(result, Var('x')))
    env[SSym('inst/alloc')] = inst_function(
        SSym('inst/alloc'), [Var('n')],
        bytecode.AllocInst(result, Var('n')))
    env[SSym('inst/load')] = inst_function(
        SSym('inst/load'), [Var('v'), Var('n')],
        bytecode.LoadInst(result, Var('v'), Var('n')))
    env[SSym('inst/store')] = inst_function(
        SSym('inst/store'), [Var('v'), Var('n'), Var('x')],
        bytecode.StoreInst(Var('v'), Var('n'), Var('x')),
        bytecode.ReturnInst(bytecode.NumLit(scheme.SNum(0))),
        should_return=False)
    env[SSym('inst/length')] = inst_function(
        SSym('inst/length'), [Var('v')],
        bytecode.LengthInst(result, Var('v')))
    env[SSym('inst/pointer=')] = inst_function(
        SSym('inst/pointer='), [Var('a'), Var('b')],
        bytecode.BinopInst(result, Binop.PTR_EQ, Var('a'), Var('b')))
    env[SSym('inst/number=')] = inst_function(
        SSym('inst/number='), [Var('a'), Var('b')],
        bytecode.BinopInst(result, Binop.NUM_EQ, Var('a'), Var('b')))
    env[SSym('inst/symbol=')] = inst_function(
        SSym('inst/symbol='), [Var('a'), Var('b')],
        bytecode.BinopInst(result, Binop.SYM_EQ, Var('a'), Var('b')))
    env[SSym('inst/number<')] = inst_function(
        SSym('inst/number<'), [Var('a'), Var('b')],
        bytecode.BinopInst(result, Binop.NUM_LT, Var('a'), Var('b')))
    env[SSym('inst/trap')] = inst_function(
        SSym('inst/trap'), [],
        bytecode.TrapInst("(trap)"),
        should_return=False)


def add_builtins(env: Dict[SSym, Value]) -> None:
    """Add builtins to the environment."""
    add_intrinsics(env)  # @TODO: Don't do this, for greater flexibility?
    code = scheme.parse("""
    (define (trap) (inst/trap))
    (define (assert b) (if b 0 (trap)))
    (define (typeof x) (inst/typeof x))

    (define (pointer= a b) (inst/pointer= a b))
    (define (symbol= a b)
      (assert (inst/symbol= (typeof a) 'symbol))
      (assert (inst/symbol= (typeof b) 'symbol))
      (inst/symbol= a b))
    (define (number= a b)
      (assert (symbol= (typeof a) 'number)
      (assert (symbol= (typeof b) 'number)
      (inst/number= a b))))

    (define (number< a b)
      (assert (symbol= (typeof a) 'number))
      (assert (symbol= (typeof b) 'number))
      (inst/number< a b))

    (define (vector-length v)
      (assert (symbol= (typeof v) 'vector))
      (inst/length v))

    (define (vector-index v n)
      (assert (symbol= (typeof v) 'vector))
      (assert (symbol= (typeof n) 'number))
      (assert (number< -1 n))
      (assert (number< n (vector-length v)))
      (inst/load v n))

    (define (vector-set! v n x)
      (assert (symbol= (typeof v) 'vector))
      (assert (symbol= (typeof n) 'number))
      (assert (number< -1 n))
      (assert (number< n (vector-length v)))
      (inst/store v n x))

    (define (vector-make/recur len idx v x)
      (vector-set! v idx x)
      (if (number= len (+ idx 1))
        v
        (vector-make/recur len (+ idx 1) v x)))

    (define (vector-make n x)
      (assert (symbol= (typeof n) 'number))
      (assert (number< -1 n))
      (if (number= n 0)
        (inst/alloc 0)
        (vector-make/recur n 0 (inst/alloc n) x)))
    """)
    emitter = FunctionEmitter(env)
    for definition in code:
        emitter.visit(definition)


def add_prelude(env: Dict[SSym, Value]) -> None:
    """Add intrinsics to the environment."""
    add_builtins(env)  # @TODO: Don't do this, for greater flexibility?
    code = scheme.parse("""
    (define (vector=/recur x y n end)
      (if (= n end)
        true
        (if (= (vector-index x n) (vector-index y n))
          (vector=/recur x y (+ n 1) end)
          false)))
    (define (vector= x y)
      (if (pointer= x y)
        true
        (if (= (vector-length x) (vector-length y))
            (vector=/recur x y 0 (vector-length x))
            false)))

    (define (= x y)
      (if (symbol= (typeof x) (typeof y))
        (if (symbol= (typeof x) 'symbol)
          (symbol= x y)
          (if (symbol= (typeof x) 'number)
            (number= x y)
            (if (symbol= (typeof x) 'vector)
              (vector= x y)
              (pointer= x y))))
        false))

    (define (< x y) (number< x y))

    (define (not b) (if b false true))
    (define (!= x y) (not (= x y)))
    (define (> x y) (< y x))
    (define (<= x y) (if (= x y) true (< x y)))
    (define (>= x y) (if (= x y) true (> x y)))

    (define (number? x) (= (typeof x) 'number))
    (define (symbol? x) (= (typeof x) 'symbol))
    (define (vector? x) (= (typeof x) 'vector))
    (define (function? x) (= (typeof x) 'function))
    (define (bool? x) (= (typeof x) 'bool))
    (define (pair? x)
      (if (vector? x)
        (= (vector-length x) 2)
        false))
    (define (nil? x) (= x []))

    (define (cons x l) [x l])
    (define (car l) (vector-index l 0))
    (define (cdr l) (vector-index l 1))
    """)
    emitter = FunctionEmitter(env)
    for definition in code:
        emitter.visit(definition)
