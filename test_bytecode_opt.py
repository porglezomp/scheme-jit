import unittest

import bytecode
from bytecode import Binop, NumLit, Var
from sexp import SNum


class OptTestCase(unittest.TestCase):
    def test_split_block(self) -> None:
        bb0 = bytecode.BasicBlock("bb0")
        bb1 = bytecode.BasicBlock("bb1")
        bb0.add_inst(bytecode.CopyInst(Var('v0'), NumLit(SNum(42))))
        bb0.add_inst(bytecode.BinopInst(
            Var('v1'), Binop.ADD, Var('v0'), NumLit(SNum(69))))
        bb0.add_inst(bytecode.JmpInst(bb1))
        bb1.add_inst(bytecode.ReturnInst(Var('v1')))
        func = bytecode.Function([], bb0)
        self.assertEqual(str(func), '''function (? ) entry=bb0
bb0:
  v0 = 42
  v1 = Binop.ADD v0 69
  jmp bb1

bb1:
  return v1''')

        bb0.split_after(0)
        self.assertEqual(str(func), '''function (? ) entry=bb0
bb0:
  v0 = 42
  jmp bb0.split

bb0.split:
  v1 = Binop.ADD v0 69
  jmp bb1

bb1:
  return v1''')
