import unittest

import bytecode
import emit_IR
import scheme


class EmitExpressionTestCase(unittest.TestCase):
    expr_emitter: emit_IR.ExpressionEmitter
    bb: bytecode.BasicBlock

    def setUp(self) -> None:
        bb_names = emit_IR.name_generator('bb')
        self.bb = bytecode.BasicBlock(next(bb_names))
        self.expr_emitter = emit_IR.ExpressionEmitter(
            self.bb, bb_names, emit_IR.name_generator('var'), {}, {})

    def test_emit_int_literal(self) -> None:
        prog = scheme.parse('42')
        self.expr_emitter.visit(prog)

        self.assertEqual(
            bytecode.NumLit(scheme.SNum(42)), self.expr_emitter.result)

    def test_emit_quoted_symbol_literal(self) -> None:
        prog = scheme.parse('spam')
        self.expr_emitter.quoted = True
        self.expr_emitter.visit(prog)

        self.assertEqual(
            bytecode.SymLit(scheme.SSym('spam')), self.expr_emitter.result)

    def test_emit_bool_literal(self) -> None:
        prog = scheme.parse('false')
        self.expr_emitter.visit(prog)

        self.assertEqual(
            bytecode.BoolLit(scheme.SBool(False)), self.expr_emitter.result)

    def test_emit_empty_vector_literal(self) -> None:
        prog = scheme.parse('[]')
        self.expr_emitter.visit(prog)

        expected_instrs = [
            bytecode.AllocInst(
                bytecode.Var('var0'),
                bytecode.NumLit(scheme.SNum(0))
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_non_empty_vector_literal(self) -> None:
        prog = scheme.parse('[1 2]')
        self.expr_emitter.visit(prog)

        arr_var = bytecode.Var('var0')
        expected_instrs = [
            bytecode.AllocInst(
                arr_var,
                bytecode.NumLit(scheme.SNum(2))
            ),
            bytecode.StoreInst(
                arr_var,
                bytecode.NumLit(scheme.SNum(0)),
                bytecode.NumLit(scheme.SNum(1))
            ),
            bytecode.StoreInst(
                arr_var,
                bytecode.NumLit(scheme.SNum(1)),
                bytecode.NumLit(scheme.SNum(2))
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_empty_quoted_list(self) -> None:
        prog = scheme.parse("'()")
        self.expr_emitter.visit(prog)

        nil_var = bytecode.Var('var0')
        expected_instrs = [
            bytecode.AllocInst(
                nil_var, bytecode.NumLit(scheme.SNum(0))
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_non_empty_quoted_list(self) -> None:
        prog = scheme.parse("'(1 spam)")
        self.expr_emitter.visit(prog)

        nil_var = bytecode.Var('var0')
        second_pair = bytecode.Var('var1')
        first_pair = bytecode.Var('var2')
        expected_instrs = [
            bytecode.AllocInst(
                nil_var, bytecode.NumLit(scheme.SNum(0))
            ),

            bytecode.AllocInst(
                second_pair, bytecode.NumLit(scheme.SNum(2))
            ),
            bytecode.StoreInst(
                second_pair,
                bytecode.NumLit(scheme.SNum(0)),
                bytecode.SymLit(scheme.SSym('spam'))
            ),
            bytecode.StoreInst(
                second_pair, bytecode.NumLit(scheme.SNum(1)), nil_var
            ),

            bytecode.AllocInst(
                first_pair, bytecode.NumLit(scheme.SNum(2))
            ),
            bytecode.StoreInst(
                first_pair,
                bytecode.NumLit(scheme.SNum(0)),
                bytecode.NumLit(scheme.SNum(1))
            ),
            bytecode.StoreInst(
                first_pair,
                bytecode.NumLit(scheme.SNum(1)),
                second_pair
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_global_function_call(self) -> None:
        prog = scheme.parse('(number? 1)')
        self.expr_emitter.global_env[scheme.SSym('number?')] = scheme.Nil
        self.expr_emitter.visit(prog)

        func_var = bytecode.Var('var0')
        result_var = bytecode.Var('var1')
        expected_instrs = [
            bytecode.LookupInst(
                func_var, bytecode.SymLit(scheme.SSym('number?'))
            ),
            bytecode.CallInst(
                result_var,
                func_var,
                [bytecode.NumLit(scheme.SNum(1))]
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_local_param_function_call(self) -> None:
        self.fail()

    def test_emit_conditional(self) -> None:
        self.fail()


class EmitFunctionDefTestCase(unittest.TestCase):
    function_emitter: emit_IR.FunctionEmitter

    def setUp(self) -> None:
        self.function_emitter = emit_IR.FunctionEmitter({})

    def test_emit_function_def(self) -> None:
        prog = scheme.parse('(define (func spam) true spam)')
        self.function_emitter.visit(prog)

        return_var = bytecode.Var('spam')
        expected = bytecode.Function(
            [return_var],
            bytecode.BasicBlock(
                'bb0',
                [bytecode.ReturnInst(return_var)]
            )
        )

        self.assertEqual(1, len(self.function_emitter.functions))
        self.assertEqual(expected, self.function_emitter.functions[0])

    def test_emit_lambda_def(self) -> None:
        prog = scheme.parse('(define (func) (lambda (spam) spam))')
        self.fail()

    def test_lambda_function_called_immediately(self) -> None:
        prog = scheme.parse('(define (func) ((lambda (spam) spam) 42))')
        self.fail()


class EmitBuiltinsTestCase(unittest.TestCase):
    pass


class TailRecursionConversionTestCase(unittest.TestCase):
    pass
