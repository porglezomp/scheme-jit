import copy
import unittest
from typing import DefaultDict, Tuple

import bytecode
import runner
import sexp
from bytecode import (BasicBlock, Binop, Function, NumLit, SymLit, TypeMap,
                      ValueMap, Var)
from optimization import FunctionOptimizer
from scheme_types import SchemeNum, SchemeObject, SchemeSym
from sexp import SFunction, SNum, SSym


def make_func() -> Function:
    bb0 = BasicBlock("bb0")
    bb1 = BasicBlock("bb1")
    bb0.add_inst(bytecode.CopyInst(Var('v0'), NumLit(SNum(42))))
    bb0.add_inst(bytecode.BinopInst(
        Var('v1'), Binop.ADD, Var('v0'), NumLit(SNum(69))))
    bb0.add_inst(bytecode.JmpInst(bb1))
    bb1.add_inst(bytecode.ReturnInst(Var('v1')))
    return Function([], bb0)


def make_branch_func_int() -> Tuple[Function, Tuple[BasicBlock, ...]]:
    bb0 = BasicBlock("bb0")
    bb1 = BasicBlock("bb1")
    bb2 = BasicBlock("bb2")
    bb3 = BasicBlock("bb3")
    bb0.add_inst(bytecode.CopyInst(Var('v0'), NumLit(SNum(42))))
    bb0.add_inst(bytecode.BrInst(Var('x'), bb1))
    bb0.add_inst(bytecode.JmpInst(bb2))
    bb1.add_inst(bytecode.BinopInst(
        Var('v0'), Binop.ADD, Var('v0'), NumLit(SNum(27))))
    bb1.add_inst(bytecode.JmpInst(bb3))
    bb2.add_inst(bytecode.BinopInst(
        Var('v0'), Binop.MUL, Var('v0'), NumLit(SNum(10))))
    bb2.add_inst(bytecode.JmpInst(bb3))
    bb3.add_inst(bytecode.ReturnInst(Var('v0')))
    return Function([Var('x')], bb0), (bb0, bb1, bb2, bb3)


def make_branch_func_object() -> Tuple[Function, Tuple[BasicBlock, ...]]:
    bb0 = BasicBlock("bb0")
    bb1 = BasicBlock("bb1")
    bb2 = BasicBlock("bb2")
    bb3 = BasicBlock("bb3")
    bb0.add_inst(bytecode.CopyInst(Var('v0'), NumLit(SNum(42))))
    bb0.add_inst(bytecode.BrInst(Var('x'), bb1))
    bb0.add_inst(bytecode.JmpInst(bb2))
    bb1.add_inst(bytecode.BinopInst(
        Var('v0'), Binop.ADD, Var('v0'), NumLit(SNum(27))))
    bb1.add_inst(bytecode.JmpInst(bb3))
    bb2.add_inst(bytecode.CopyInst(Var('v0'), SymLit(SSym('hi'))))
    bb2.add_inst(bytecode.JmpInst(bb3))
    bb3.add_inst(bytecode.ReturnInst(Var('v0')))
    return Function([Var('x')], bb0), (bb0, bb1, bb2, bb3)


def get_builtins() -> bytecode.EvalEnv:
    env = bytecode.EvalEnv(bytecode_jit=True)
    runner.add_intrinsics(env)
    runner.add_builtins(env)
    return env


class OptTestCase(unittest.TestCase):
    def test_baseline(self) -> None:
        self.assertEqual(str(make_func()), '''function (?) entry=bb0
bb0:
  v0 = 42
  v1 = Binop.ADD v0 69
  jmp bb1

bb1:
  return v1''')

    def test_split_block(self) -> None:
        func = make_func()
        func.start.split_after(0)
        self.assertEqual(str(func), '''function (?) entry=bb0
bb0:
  v0 = 42
  jmp bb0.split0

bb0.split0:
  v1 = Binop.ADD v0 69
  jmp bb1

bb1:
  return v1''')

    def test_mark_vars(self) -> None:
        func = make_func()
        opt = FunctionOptimizer(func)
        func = opt.mark_vars(func)
        self.assertEqual(str(func), '''function (?) entry=inl0@bb0
inl0@bb0:
  inl0@v0 = 42
  inl0@v1 = Binop.ADD inl0@v0 69
  jmp inl0@bb1

inl0@bb1:
  return inl0@v1''')

    def test_mark_functions(self) -> None:
        func = make_func()
        func.start.split_after(0)
        bb0 = func.start
        bb0_split = next(bb0.successors())
        bb1 = next(bb0_split.successors())

        opt = FunctionOptimizer(func)
        opt.compute_preds()
        self.assertEqual(opt.succs, DefaultDict(list, {
            id(bb0): [(bb0, 1, bb0_split)],
            id(bb0_split): [(bb0_split, 1, bb1)],
            id(bb1): [],
        }))
        self.assertEqual(opt.preds, DefaultDict(list, {
            id(bb0): [],
            id(bb0_split): [(bb0, 1, bb0_split)],
            id(bb1): [(bb0_split, 1, bb1)],
        }))

    def test_block_transfer(self) -> None:
        func = make_func()
        opt = FunctionOptimizer(func)
        data = opt.block_transfer(
            bytecode.EvalEnv(), func.start, TypeMap(), ValueMap())
        self.assertEqual(data, [
            (TypeMap(), ValueMap()),
            (TypeMap({Var('v0'): SchemeNum}),
             ValueMap({Var('v0'): SNum(42)})),
            (TypeMap({Var('v0'): SchemeNum, Var('v1'): SchemeNum}),
             ValueMap({Var('v0'): SNum(42), Var('v1'): SNum(111)})),
            (TypeMap({Var('v0'): SchemeNum, Var('v1'): SchemeNum}),
             ValueMap({Var('v0'): SNum(42), Var('v1'): SNum(111)})),
        ])

    def test_dataflow_stable_type(self) -> None:
        func, blocks = make_branch_func_int()
        bb0, bb1, bb2, bb3 = blocks
        opt = FunctionOptimizer(func)
        opt.dataflow(bytecode.EvalEnv())
        after_0 = (TypeMap({Var('v0'): SchemeNum}),
                   ValueMap({Var('v0'): SNum(42)}))
        self.assertEqual(opt.block_input_maps(bb1), after_0)
        self.assertEqual(opt.block_input_maps(bb2), after_0)
        self.assertEqual(
            opt.block_input_maps(bb3),
            (TypeMap({Var('v0'): SchemeNum}), ValueMap()),
        )
        self.assertEqual(opt.info, {
            id(bb0): [(TypeMap(), ValueMap()), after_0, after_0, after_0],
            id(bb1): [
                after_0,
                (TypeMap({Var('v0'): SchemeNum}),
                 ValueMap({Var('v0'): SNum(69)})),
                (TypeMap({Var('v0'): SchemeNum}),
                 ValueMap({Var('v0'): SNum(69)})),
            ],
            id(bb2): [
                after_0,
                (TypeMap({Var('v0'): SchemeNum}),
                 ValueMap({Var('v0'): SNum(420)})),
                (TypeMap({Var('v0'): SchemeNum}),
                 ValueMap({Var('v0'): SNum(420)})),
            ],
            id(bb3): [
                (TypeMap({Var('v0'): SchemeNum}), ValueMap()),
                (TypeMap({Var('v0'): SchemeNum}), ValueMap()),
            ],
        })

    def test_dataflow_unstable_type(self) -> None:
        func, blocks = make_branch_func_object()
        bb0, bb1, bb2, bb3 = blocks
        opt = FunctionOptimizer(func)
        opt.dataflow(bytecode.EvalEnv())
        after_0 = (TypeMap({Var('v0'): SchemeNum}),
                   ValueMap({Var('v0'): SNum(42)}))
        self.assertEqual(opt.block_input_maps(bb1), after_0)
        self.assertEqual(opt.block_input_maps(bb2), after_0)
        self.assertEqual(
            opt.block_input_maps(bb3),
            (TypeMap({Var('v0'): SchemeObject}), ValueMap()),
        )
        self.assertEqual(opt.info, {
            id(bb0): [(TypeMap(), ValueMap()), after_0, after_0, after_0],
            id(bb1): [
                after_0,
                (TypeMap({Var('v0'): SchemeNum}),
                 ValueMap({Var('v0'): SNum(69)})),
                (TypeMap({Var('v0'): SchemeNum}),
                 ValueMap({Var('v0'): SNum(69)})),
            ],
            id(bb2): [
                after_0,
                (TypeMap({Var('v0'): SchemeSym}),
                 ValueMap({Var('v0'): SSym('hi')})),
                (TypeMap({Var('v0'): SchemeSym}),
                 ValueMap({Var('v0'): SSym('hi')})),
            ],
            id(bb3): [
                (TypeMap({Var('v0'): SchemeObject}), ValueMap()),
                (TypeMap({Var('v0'): SchemeObject}), ValueMap()),
            ],
        })

    def test_optimize(self) -> None:
        env = get_builtins()
        func = env._global_env[SSym('+')]
        assert isinstance(func, sexp.SFunction)
        assert func.code

        code = copy.deepcopy(func.code)
        self.assertEqual(str(code), '''
function (? a b) entry=bb0
bb0:
  inl4@inl1@inl0@result = typeof a
  inl4@inl0@inl0@result = Binop.SYM_EQ inl4@inl1@inl0@result 'number
  brn inl4@inl0@inl0@result bb0.split5
  inl2@inl1@inl0@result = typeof b
  inl2@inl0@inl0@result = Binop.SYM_EQ inl2@inl1@inl0@result 'number
  brn inl2@inl0@inl0@result bb0.split6
  inl0@result = Binop.ADD a b
  return inl0@result

bb0.split5:
  trap '(trap)'

bb0.split6:
  trap '(trap)'
'''.strip())

        opt = FunctionOptimizer(code)
        opt.specialization = (SchemeNum, SchemeNum)
        opt.optimize(env)

        self.assertEqual(str(code), '''
function (? a b) entry=bb0
bb0:
  inl0@result = Binop.ADD a b
  return inl0@result
'''.strip())

    def test_specialize_value(self) -> None:
        env = get_builtins()
        func = env._global_env[SSym('+')]
        assert isinstance(func, sexp.SFunction)
        assert func.code

        code = copy.deepcopy(func.code)
        opt = FunctionOptimizer(code)
        opt.specialization = (SchemeNum, SchemeNum)
        opt.inputs = (SNum(42), SNum(27))
        opt.optimize(env)

        self.assertEqual(str(code), '''
function (? a b) entry=bb0
bb0:
  return 69
        '''.strip())
