import unittest

import bytecode
from bytecode import Var, SymLit
from scheme import SSym


class BytecodeTestCast(unittest.TestCase):
    def test_example_recursive(self) -> None:
        """
        function list? (v0) entry=bb0
        bb0:
            v1 = lookup 'nil?
            v2 = call v1 (v0)
            br v2 bb1 bb2

        bb1:
            result = 'true
            jmp end

        bb2:
            v3 = lookup 'pair?
            v4 = call v3 (v0)
            br v4 bb3 bb4

        bb3:
            v5 = lookup 'cdr
            v6 = call v5 (v0)
            v7 = lookup 'list?
            result = call v7 (v6)
            jmp end

        bb4:
            result = 'false
            jmp end

        end:
            return result
        """
        result = Var("result")
        bb0 = bytecode.BasicBlock("bb0")
        bb1 = bytecode.BasicBlock("bb1")
        bb2 = bytecode.BasicBlock("bb2")
        bb3 = bytecode.BasicBlock("bb3")
        bb4 = bytecode.BasicBlock("bb4")
        end = bytecode.ReturnBlock("end", result)
        is_list = bytecode.Function(params=[Var("v0")], start=bb0, finish=end)

        bb0.add_inst(bytecode.LookupInst(Var("v1"), SymLit(SSym("nil?"))))
        bb0.add_inst(bytecode.CallInst(Var("v2"), Var("v1"), [Var("v1")]))
        bb0.terminator = bytecode.Br(Var("v2"), bb1, bb2)

        bb1.add_inst(bytecode.CopyInst(result, SymLit(SSym("true"))))
        bb1.terminator = bytecode.Jmp(end)

        bb2.add_inst(bytecode.LookupInst(Var("v3"), SymLit(SSym("pair?"))))
        bb2.add_inst(bytecode.CallInst(Var("v4"), Var("v3"), [Var("v0")]))
        bb2.terminator = bytecode.Br(Var("v4"), bb3, bb4)

        bb3.add_inst(bytecode.LookupInst(Var("v5"), SymLit(SSym("cdr"))))
        bb3.add_inst(bytecode.CallInst(Var("v6"), Var("v5"), [Var("v0")]))
        bb3.add_inst(bytecode.LookupInst(Var("v7"), SymLit(SSym("list?"))))
        bb3.add_inst(bytecode.CallInst(result, Var("v7"), [Var("v6")]))
        bb3.terminator = bytecode.Jmp(end)

        bb4.add_inst(bytecode.CopyInst(result, SymLit(SSym("false"))))
        bb4.terminator = bytecode.Jmp(end)

        print(is_list)

    def test_example_tail_call(self) -> None:
        """
        function list? (v0) entry=bb0
        bb0:
            v1 = lookup 'nil?
            v2 = call v1 (v0)
            br v2 bb1 bb2

        bb1:
            result = 'true
            jmp end

        bb2:
            v3 = lookup 'pair?
            v4 = call v3 (v0)
            br v4 bb3 bb4

        bb3:
            v5 = lookup 'cdr
            v0 = call v5 (v0)
            jmp bb0

        bb4:
            result = 'false
            jmp end

        end:
            return result
        """
        result = Var("result")
        bb0 = bytecode.BasicBlock("bb0")
        bb1 = bytecode.BasicBlock("bb1")
        bb2 = bytecode.BasicBlock("bb2")
        bb3 = bytecode.BasicBlock("bb3")
        bb4 = bytecode.BasicBlock("bb4")
        end = bytecode.ReturnBlock("end", result)
        is_list = bytecode.Function(params=[Var("v0")], start=bb0, finish=end)

        bb0.add_inst(bytecode.LookupInst(Var("v1"), SymLit(SSym("nil?"))))
        bb0.add_inst(bytecode.CallInst(Var("v2"), Var("v1"), [Var("v0")]))
        bb0.terminator = bytecode.Br(Var("v2"), bb1, bb2)

        bb1.add_inst(bytecode.CopyInst(result, SymLit(SSym("true"))))
        bb1.terminator = bytecode.Jmp(end)

        bb2.add_inst(bytecode.LookupInst(Var("v3"), SymLit(SSym("pair?"))))
        bb2.add_inst(bytecode.CallInst(Var("v4"), Var("v3"), [Var("v0")]))
        bb2.terminator = bytecode.Br(Var("v4"), bb3, bb4)

        bb3.add_inst(bytecode.LookupInst(Var("v5"), SymLit(SSym("global"))))
        bb3.add_inst(bytecode.CallInst(Var("v0"), Var("v5"), [Var("v0")]))
        bb3.terminator = bytecode.Jmp(bb0)

        bb4.add_inst(bytecode.CopyInst(result, SymLit(SSym("false"))))
        bb4.terminator = bytecode.Jmp(end)

        # These tests are just to test the API
        print(is_list)
