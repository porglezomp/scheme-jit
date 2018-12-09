import unittest

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
    def test_split_block(self) -> None:
        func = make_func()
        self.assertEqual(str(func), '''function (?) entry=bb0
bb0:
  v0 = 42
  v1 = Binop.ADD v0 69
  jmp bb1

bb1:
  return v1''')

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
        opt = FunctionOptimizer()
        func = make_func()
        self.assertEqual(str(func), '''function (?) entry=bb0
bb0:
  v0 = 42
  v1 = Binop.ADD v0 69
  jmp bb1

bb1:
  return v1''')

        func = opt.mark_vars(func)
        self.assertEqual(str(func), '''function (?) entry=inl0@bb0
inl0@bb0:
  inl0@v0 = 42
  inl0@v1 = Binop.ADD inl0@v0 69
  jmp inl0@bb1

inl0@bb1:
  return inl0@v1''')
