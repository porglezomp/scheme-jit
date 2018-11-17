from typing import Dict, List

import bytecode
import scheme
from bytecode import BasicBlock, Binop, Function, Inst, Var
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
    env[SSym('typeof')] = inst_function(
        SSym('typeof'), [Var('x')],
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
    env[SSym('trap')] = inst_function(
        SSym('trap'), [],
        bytecode.TrapInst("(trap)"),
        should_return=False)


def add_builtins(env: Dict[SSym, Value]) -> None:
    """Add builtins to the environment."""
    add_intrinsics(env)  # @TODO: Don't do this, for greater flexibility?
    raise NotImplementedError("builtins")


def add_prelude(env: Dict[SSym, Value]) -> None:
    """Add intrinsics to the environment."""
    add_builtins(env)  # @TODO: Don't do this, for greater flexibility?
    raise NotImplementedError("Prelude")
