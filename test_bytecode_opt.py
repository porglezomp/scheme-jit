import unittest
from typing import DefaultDict

import bytecode
from bytecode import Binop, NumLit, Var
from optimization import FunctionOptimizer
from sexp import SNum


def make_func() -> bytecode.Function:
    bb0 = bytecode.BasicBlock("bb0")
    bb1 = bytecode.BasicBlock("bb1")
    bb0.add_inst(bytecode.CopyInst(Var('v0'), NumLit(SNum(42))))
    bb0.add_inst(bytecode.BinopInst(
        Var('v1'), Binop.ADD, Var('v0'), NumLit(SNum(69))))
    bb0.add_inst(bytecode.JmpInst(bb1))
    bb1.add_inst(bytecode.ReturnInst(Var('v1')))
    return bytecode.Function([], bb0)


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
  jmp bb0.split

bb0.split:
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
