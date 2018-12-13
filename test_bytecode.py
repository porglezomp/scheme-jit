
import unittest
from typing import Any

import bytecode
import scheme_types
import sexp
from bytecode import Binop, BoolLit, NumLit, SymLit, Var
from sexp import SBool, SNum, SSym, SVect


class BytecodeTestCase(unittest.TestCase):
    def test_example_recursive(self) -> None:
        """
        function list? (v0) entry=bb0
        bb0:
            v1 = lookup 'nil?
            v2 = call v1 (v0)
            br v2 bb1
            v3 = lookup 'pair?
            v4 = call v3 (v0)
            brn v4 bb2
            v5 = lookup 'cdr
            v6 = call v5 (v0)
            v7 = lookup 'list?
            v8 = call v7 (v6)
            return v8

        bb1:
            return 'true

        bb2:
            return 'false
        """
        bb0 = bytecode.BasicBlock("bb0")
        bb1 = bytecode.BasicBlock("bb1")
        bb2 = bytecode.BasicBlock("bb2")
        is_list = bytecode.Function([Var("v0")], bb0)

        bb0.add_inst(bytecode.LookupInst(Var("v1"), SymLit(SSym("nil?"))))
        bb0.add_inst(bytecode.CallInst(Var("v2"), Var("v1"), [Var("v1")]))
        bb0.add_inst(bytecode.BrInst(Var("v2"), bb1))
        bb0.add_inst(bytecode.LookupInst(Var("v3"), SymLit(SSym("pair?"))))
        bb0.add_inst(bytecode.CallInst(Var("v4"), Var("v3"), [Var("v0")]))
        bb0.add_inst(bytecode.BrnInst(Var("v4"), bb2))
        bb0.add_inst(bytecode.LookupInst(Var("v5"), SymLit(SSym("cdr"))))
        bb0.add_inst(bytecode.CallInst(Var("v6"), Var("v5"), [Var("v0")]))
        bb0.add_inst(bytecode.LookupInst(Var("v7"), SymLit(SSym("list?"))))
        bb0.add_inst(bytecode.CallInst(Var("v8"), Var("v7"), [Var("v6")]))
        bb0.add_inst(bytecode.ReturnInst(Var("v8")))

        bb1.add_inst(bytecode.ReturnInst(SymLit(SSym("true"))))

        bb2.add_inst(bytecode.ReturnInst(SymLit(SSym("false"))))

        # These tests are just to test the API
        assert is_list

    def test_example_tail_call(self) -> None:
        """
        function list? (v0) entry=bb0
        bb0:
            v1 = lookup 'nil?
            v2 = call v1 (v0)
            br v2 bb1
            v3 = lookup 'pair?
            v4 = call v3 (v0)
            brn v4 bb2
            v5 = lookup 'cdr
            v0 = call v5 (v0)
            jmp bb0

        bb1:
            return 'true

        bb2:
            return 'false
        """
        bb0 = bytecode.BasicBlock("bb0")
        bb1 = bytecode.BasicBlock("bb1")
        bb2 = bytecode.BasicBlock("bb2")
        is_list = bytecode.Function([Var("v0")], bb0)

        bb0.add_inst(bytecode.LookupInst(Var("v1"), SymLit(SSym("nil?"))))
        bb0.add_inst(bytecode.CallInst(Var("v2"), Var("v1"), [Var("v0")]))
        bb0.add_inst(bytecode.BrInst(Var("v2"), bb1))
        bb0.add_inst(bytecode.LookupInst(Var("v3"), SymLit(SSym("pair?"))))
        bb0.add_inst(bytecode.CallInst(Var("v4"), Var("v3"), [Var("v0")]))
        bb0.add_inst(bytecode.BrnInst(Var("v4"), bb2))
        bb0.add_inst(bytecode.LookupInst(Var("v5"), SymLit(SSym("global"))))
        bb0.add_inst(bytecode.CallInst(Var("v0"), Var("v5"), [Var("v0")]))
        bb0.add_inst(bytecode.JmpInst(bb0))

        bb1.add_inst(bytecode.ReturnInst(SymLit(SSym("true"))))

        bb2.add_inst(bytecode.ReturnInst(SymLit(SSym("false"))))

        assert is_list

    def test_example_inlined(self) -> None:
        """
        function list? (v0) entry=bb0
        bb0:
            v1 = typeof v0
            v2 = sym_eq v1 'vector
            brn v2 false
            v3 = length v0
            v4 = num_eq v3 0
            br v4 true
            v5 = num_eq v3 2
            brn v5 false
            v0 = load v0 [1]
            jmp bb0

        true:
            return 'true

        false:
            return 'false
        """
        bb0 = bytecode.BasicBlock("bb0")
        true = bytecode.BasicBlock("true")
        false = bytecode.BasicBlock("false")
        is_list = bytecode.Function([Var("v0")], bb0)

        # These tests are just to test the API
        bb0.add_inst(bytecode.TypeofInst(Var("v1"), Var("v0")))
        bb0.add_inst(bytecode.BinopInst(
            Var("v2"), Binop.SYM_EQ, Var("v1"), SymLit(SSym("vector"))))
        bb0.add_inst(bytecode.BrnInst(Var("v2"), false))
        bb0.add_inst(bytecode.LengthInst(Var("v3"), Var("v0")))
        bb0.add_inst(bytecode.BinopInst(
            Var("v4"), Binop.NUM_EQ, Var("v3"), NumLit(SNum(0))))
        bb0.add_inst(bytecode.BrInst(Var("v4"), true))

        bb0.add_inst(bytecode.BinopInst(
            Var("v5"), Binop.NUM_EQ, Var("v3"), NumLit(SNum(2))))
        bb0.add_inst(bytecode.BrnInst(Var("v5"), false))
        bb0.add_inst(bytecode.LoadInst(Var("v0"), Var("v0"), NumLit(SNum(1))))
        bb0.add_inst(bytecode.JmpInst(bb0))

        true.add_inst(bytecode.ReturnInst(BoolLit(SBool(True))))

        false.add_inst(bytecode.ReturnInst(BoolLit(SBool(False))))

        env = bytecode.EvalEnv(local_env={Var("v0"): SNum(42)})
        gen = bytecode.ResultGenerator(is_list.run(env))
        gen.run()
        self.assertEqual(gen.value, SBool(False))

        env = bytecode.EvalEnv(local_env={
            Var("v0"): SVect([SNum(42), SVect([SNum(69), SVect([])])])
        })
        gen = bytecode.ResultGenerator(is_list.run(env))
        gen.run()
        self.assertEqual(gen.value, SBool(True))

    def test_call_specialized(self) -> None:
        bb0 = bytecode.BasicBlock("bb0")
        bb0.add_inst(bytecode.ReturnInst(BoolLit(SBool(False))))
        byte_func = bytecode.Function([Var("x")], bb0)
        bb0_specialized = bytecode.BasicBlock("bb0")
        bb0_specialized.add_inst(bytecode.ReturnInst(BoolLit(SBool(True))))
        byte_func_specialized = bytecode.Function([Var("x")], bb0_specialized)

        func = sexp.SFunction(
            SSym("func"), [SSym("x")], sexp.to_slist([]),
            code=byte_func, is_lambda=False,
            specializations={
                (scheme_types.SchemeSym,): byte_func_specialized
            },
        )

        env = bytecode.EvalEnv(local_env={Var('f'): func})
        bytecode.CallInst(Var('y'), Var('f'), [NumLit(SNum(42))]).run(env)
        assert env[Var('y')] == SBool(False)
        bytecode.CallInst(Var('y'), Var('f'), [SymLit(SSym('x'))]).run(env)
        assert env[Var('y')] == SBool(True)
        bytecode.CallInst(Var('y'), Var('f'), [SymLit(SSym('x'))],
                          specialization=(scheme_types.SchemeSym,)).run(env)
        assert env[Var('y')] == SBool(True)

        # If specialization not found, fall back to dynamic dispatch
        bytecode.CallInst(
            Var('y'), Var('f'), [SymLit(SSym('x'))],
            specialization=(scheme_types.SchemeNum,)).run(env)
        self.assertEqual(env[Var('y')], SBool(True))

        bytecode.CallInst(
            Var('y'), Var('f'), [NumLit(SNum(42))],
            specialization=(scheme_types.SchemeNum,)).run(env)
        self.assertEqual(env[Var('y')], SBool(False))
