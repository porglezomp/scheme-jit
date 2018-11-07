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
        expected_instrs = [
            bytecode.LookupInst(
                func_var, bytecode.SymLit(scheme.SSym('number?'))
            ),
            bytecode.CallInst(
                bytecode.Var('var1'),
                func_var,
                [bytecode.NumLit(scheme.SNum(1))]
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_local_var_function_call(self) -> None:
        prog = scheme.parse('(local_var 42)')
        lambda_var = bytecode.Var('local_var')
        self.expr_emitter.local_env[scheme.SSym('local_var')] = lambda_var
        self.expr_emitter.visit(prog)

        expected_instrs = [
            bytecode.CallInst(
                bytecode.Var('var0'),
                lambda_var,
                [bytecode.NumLit(scheme.SNum(42))]
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_lambda_function_called_immediately(self) -> None:
        prog = scheme.parse('((lambda (spam) spam) 42)')
        self.expr_emitter.visit(prog)

        expected_lambda = bytecode.Function(
            [bytecode.Var('spam')],
            bytecode.BasicBlock(
                'bb0',
                [
                    bytecode.ReturnInst(bytecode.Var('spam'))
                ]
            )
        )

        lambda_lookup_var = bytecode.Var('var0')
        expected_instrs = [
            bytecode.LookupInst(
                lambda_lookup_var, bytecode.SymLit(scheme.SSym('__lambda0'))
            ),
            bytecode.CallInst(
                bytecode.Var('var1'),
                lambda_lookup_var,
                [bytecode.NumLit(scheme.SNum(42))]
            )
        ]

        actual_lambda = (
            self.expr_emitter.global_env[scheme.SSym('__lambda0')])
        assert isinstance(actual_lambda, scheme.SFunction)
        self.assertEqual(expected_lambda, actual_lambda.code)

        self.assertEqual(expected_instrs, self.bb.instructions)

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

        self.assertEqual(1, len(self.function_emitter.global_env))
        actual_func = self.function_emitter.global_env[scheme.SSym('func')]
        assert isinstance(actual_func, scheme.SFunction)
        self.assertEqual(expected, actual_func.code)

    def test_emit_multiple_function_defs(self) -> None:
        prog = scheme.parse('(define (func) 42) (define (func2) 43)')
        self.function_emitter.visit(prog)

        expected_func = bytecode.Function(
            [],
            bytecode.BasicBlock(
                'bb0',
                [bytecode.ReturnInst(bytecode.NumLit(scheme.SNum(42)))]
            )
        )

        expected_func2 = bytecode.Function(
            [],
            bytecode.BasicBlock(
                'bb0',
                [bytecode.ReturnInst(bytecode.NumLit(scheme.SNum(43)))]
            )
        )

        self.assertEqual(2, len(self.function_emitter.global_env))

        actual_func = self.function_emitter.global_env[scheme.SSym('func')]
        assert isinstance(actual_func, scheme.SFunction)
        self.assertEqual(expected_func, actual_func.code)

        actual_func2 = self.function_emitter.global_env[scheme.SSym('func2')]
        assert isinstance(actual_func2, scheme.SFunction)
        self.assertEqual(expected_func2, actual_func2.code)

    def test_emit_lambda_def(self) -> None:
        prog = scheme.parse('(define (func) (lambda (spam) spam))')
        self.function_emitter.visit(prog)

        func_ret_var = bytecode.Var('var0')
        expected_func = bytecode.Function(
            [],
            bytecode.BasicBlock(
                'bb0',
                [
                    bytecode.LookupInst(
                        func_ret_var, bytecode.SymLit(scheme.SSym('__lambda0'))
                    ),
                    bytecode.ReturnInst(func_ret_var)
                ]
            )
        )

        expected_lambda = bytecode.Function(
            [bytecode.Var('spam')],
            bytecode.BasicBlock(
                'bb0',
                [
                    bytecode.ReturnInst(bytecode.Var('spam'))
                ]
            )
        )

        actual_func = self.function_emitter.global_env[scheme.SSym('func')]
        assert isinstance(actual_func, scheme.SFunction)
        self.assertEqual(expected_func, actual_func.code)

        actual_lambda = (
            self.function_emitter.global_env[scheme.SSym('__lambda0')])
        assert isinstance(actual_lambda, scheme.SFunction)
        self.assertEqual(expected_lambda, actual_lambda.code)


class EmitBuiltinsTestCase(unittest.TestCase):
    pass


class TailRecursionConversionTestCase(unittest.TestCase):
    pass
