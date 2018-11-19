import unittest

import bytecode
import emit_IR
import sexp


class EmitExpressionTestCase(unittest.TestCase):
    expr_emitter: emit_IR.ExpressionEmitter
    bb: bytecode.BasicBlock

    def setUp(self) -> None:
        bb_names = emit_IR.name_generator('bb')
        self.bb = bytecode.BasicBlock(next(bb_names))
        self.expr_emitter = emit_IR.ExpressionEmitter(
            self.bb, bb_names, emit_IR.name_generator('var'), {}, {})

    def test_emit_int_literal(self) -> None:
        prog = sexp.parse('42')
        self.expr_emitter.visit(prog)

        self.assertEqual(
            bytecode.NumLit(sexp.SNum(42)), self.expr_emitter.result)

    def test_emit_quoted_symbol_literal(self) -> None:
        prog = sexp.parse("'spam")
        self.expr_emitter.visit(prog)

        self.assertEqual(
            bytecode.SymLit(sexp.SSym('spam')), self.expr_emitter.result)

    def test_emit_bool_literal(self) -> None:
        prog = sexp.parse('false')
        self.expr_emitter.visit(prog)

        self.assertEqual(
            bytecode.BoolLit(sexp.SBool(False)), self.expr_emitter.result)

    def test_emit_empty_vector_literal(self) -> None:
        prog = sexp.parse('[]')
        self.expr_emitter.visit(prog)

        expected_instrs = [
            bytecode.AllocInst(
                bytecode.Var('var0'),
                bytecode.NumLit(sexp.SNum(0))
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_non_empty_vector_literal(self) -> None:
        prog = sexp.parse('[1 2]')
        self.expr_emitter.visit(prog)

        arr_var = bytecode.Var('var0')
        expected_instrs = [
            bytecode.AllocInst(
                arr_var,
                bytecode.NumLit(sexp.SNum(2))
            ),
            bytecode.StoreInst(
                arr_var,
                bytecode.NumLit(sexp.SNum(0)),
                bytecode.NumLit(sexp.SNum(1))
            ),
            bytecode.StoreInst(
                arr_var,
                bytecode.NumLit(sexp.SNum(1)),
                bytecode.NumLit(sexp.SNum(2))
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_empty_quoted_list(self) -> None:
        prog = sexp.parse("'()")
        self.expr_emitter.visit(prog)

        nil_var = bytecode.Var('var0')
        expected_instrs = [
            bytecode.AllocInst(
                nil_var, bytecode.NumLit(sexp.SNum(0))
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_non_empty_quoted_list(self) -> None:
        prog = sexp.parse("'(1 spam)")
        self.expr_emitter.visit(prog)

        nil_var = bytecode.Var('var0')
        second_pair = bytecode.Var('var1')
        first_pair = bytecode.Var('var2')
        expected_instrs = [
            bytecode.AllocInst(
                nil_var, bytecode.NumLit(sexp.SNum(0))
            ),

            bytecode.AllocInst(
                second_pair, bytecode.NumLit(sexp.SNum(2))
            ),
            bytecode.StoreInst(
                second_pair,
                bytecode.NumLit(sexp.SNum(0)),
                bytecode.SymLit(sexp.SSym('spam'))
            ),
            bytecode.StoreInst(
                second_pair, bytecode.NumLit(sexp.SNum(1)), nil_var
            ),

            bytecode.AllocInst(
                first_pair, bytecode.NumLit(sexp.SNum(2))
            ),
            bytecode.StoreInst(
                first_pair,
                bytecode.NumLit(sexp.SNum(0)),
                bytecode.NumLit(sexp.SNum(1))
            ),
            bytecode.StoreInst(
                first_pair,
                bytecode.NumLit(sexp.SNum(1)),
                second_pair
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_global_function_call(self) -> None:
        prog = sexp.parse('(number? 1)')
        self.expr_emitter.visit(prog)

        func_var = bytecode.Var('var0')
        typeof_var = bytecode.Var('__typeof')
        is_function_var = bytecode.Var('__is_func')
        expected_instrs = [
            bytecode.LookupInst(
                func_var, bytecode.SymLit(sexp.SSym('number?'))
            ),

            bytecode.TypeofInst(typeof_var, func_var),
            bytecode.BinopInst(
                is_function_var, bytecode.Binop.SYM_EQ,
                typeof_var, bytecode.SymLit(sexp.SSym('function'))
            ),

            bytecode.BrnInst(
                is_function_var,
                bytecode.BasicBlock(
                    '__non_function_trap',
                    [bytecode.TrapInst('Attempted to call a non-function')])
            ),

            bytecode.CallInst(
                bytecode.Var('var1'),
                func_var,
                [bytecode.NumLit(sexp.SNum(1))]
            ),
        ]
        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_local_var_function_call(self) -> None:
        prog = sexp.parse('(local_var 42)')
        lambda_var = bytecode.Var('local_var')
        self.expr_emitter.local_env[sexp.SSym('local_var')] = lambda_var
        self.expr_emitter.visit(prog)

        typeof_var = bytecode.Var('__typeof')
        is_function_var = bytecode.Var('__is_func')
        expected_instrs = [
            bytecode.TypeofInst(typeof_var, lambda_var),
            bytecode.BinopInst(
                is_function_var, bytecode.Binop.SYM_EQ,
                typeof_var, bytecode.SymLit(sexp.SSym('function'))
            ),
            bytecode.BrnInst(
                is_function_var,
                bytecode.BasicBlock(
                    '__non_function_trap',
                    [bytecode.TrapInst('Attempted to call a non-function')])
            ),

            bytecode.CallInst(
                bytecode.Var('var0'),
                lambda_var,
                [bytecode.NumLit(sexp.SNum(42))]
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_lambda_function_called_immediately(self) -> None:
        prog = sexp.parse('((lambda (spam) spam) 42)')
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
        typeof_var = bytecode.Var('__typeof')
        is_function_var = bytecode.Var('__is_func')
        expected_instrs = [
            bytecode.LookupInst(
                lambda_lookup_var, bytecode.SymLit(sexp.SSym('__lambda0'))
            ),

            bytecode.TypeofInst(typeof_var, lambda_lookup_var),
            bytecode.BinopInst(
                is_function_var, bytecode.Binop.SYM_EQ,
                typeof_var, bytecode.SymLit(sexp.SSym('function'))
            ),
            bytecode.BrnInst(
                is_function_var,
                bytecode.BasicBlock(
                    '__non_function_trap',
                    [bytecode.TrapInst('Attempted to call a non-function')])
            ),

            bytecode.CallInst(
                bytecode.Var('var1'),
                lambda_lookup_var,
                [bytecode.NumLit(sexp.SNum(42))]
            )
        ]

        actual_lambda = (
            self.expr_emitter.global_env[sexp.SSym('__lambda0')])
        assert isinstance(actual_lambda, sexp.SFunction)
        self.assertEqual(expected_lambda, actual_lambda.code)

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_conditional(self) -> None:
        prog = sexp.parse('(if true 42 43)')
        self.expr_emitter.visit(prog)

        result_var = bytecode.Var('var0')
        self.assertEqual(result_var, self.expr_emitter.result)
        self.assertIsNot(
            self.expr_emitter.parent_block, self.expr_emitter.end_block)

        expected_end_block = bytecode.BasicBlock('bb3')

        expected_else_block = bytecode.BasicBlock(
            'bb2',
            [
                bytecode.CopyInst(
                    result_var, bytecode.NumLit(sexp.SNum(43))
                ),
                bytecode.JmpInst(expected_end_block)
            ]
        )

        expected_then_block = bytecode.BasicBlock(
            'bb1',
            [
                bytecode.CopyInst(
                    result_var, bytecode.NumLit(sexp.SNum(42))
                ),
                bytecode.JmpInst(expected_end_block)
            ]
        )

        expected_parent_block = bytecode.BasicBlock(
            'bb0',
            [
                bytecode.BrInst(
                    bytecode.BoolLit(sexp.SBool(True)),
                    expected_then_block
                ),
                bytecode.JmpInst(expected_else_block)
            ]
        )

        self.assertEqual(expected_parent_block, self.expr_emitter.parent_block)
        self.assertEqual(expected_end_block, self.expr_emitter.end_block)

    def test_nested_conditionals(self) -> None:
        prog = sexp.parse('(if true (if false 42 43) (if true 44 45))')
        self.expr_emitter.visit(prog)

        outer_result_var = bytecode.Var('var0')

        outer_end_block = bytecode.BasicBlock('bb9')

        then_result_var = bytecode.Var('var1')
        then_end_block = bytecode.BasicBlock(
            'bb5',
            [
                bytecode.CopyInst(outer_result_var, then_result_var),
                bytecode.JmpInst(outer_end_block)
            ]
        )
        then_then_block = bytecode.BasicBlock(
            'bb3',
            [
                bytecode.CopyInst(
                    then_result_var, bytecode.NumLit(sexp.SNum(42))
                ),
                bytecode.JmpInst(then_end_block)
            ]
        )
        then_else_block = bytecode.BasicBlock(
            'bb4',
            [
                bytecode.CopyInst(
                    then_result_var, bytecode.NumLit(sexp.SNum(43))
                ),
                bytecode.JmpInst(then_end_block)
            ]
        )

        else_result_var = bytecode.Var('var2')
        else_end_block = bytecode.BasicBlock(
            'bb8',
            [
                bytecode.CopyInst(outer_result_var, else_result_var),
                bytecode.JmpInst(outer_end_block)
            ]
        )
        else_then_block = bytecode.BasicBlock(
            'bb6',
            [
                bytecode.CopyInst(
                    else_result_var, bytecode.NumLit(sexp.SNum(44))
                ),
                bytecode.JmpInst(else_end_block)
            ]
        )
        else_else_block = bytecode.BasicBlock(
            'bb7',
            [
                bytecode.CopyInst(
                    else_result_var, bytecode.NumLit(sexp.SNum(45))
                ),
                bytecode.JmpInst(else_end_block)
            ]
        )

        then_block = bytecode.BasicBlock(
            'bb1',
            [
                bytecode.BrInst(
                    bytecode.BoolLit(sexp.SBool(False)), then_then_block
                ),
                bytecode.JmpInst(then_else_block)
            ]
        )
        else_block = bytecode.BasicBlock(
            'bb2',
            [
                bytecode.BrInst(
                    bytecode.BoolLit(sexp.SBool(True)), else_then_block
                ),
                bytecode.JmpInst(else_else_block)
            ]
        )

        entry_block = bytecode.BasicBlock(
            'bb0',
            [
                bytecode.BrInst(
                    bytecode.BoolLit(sexp.SBool(True)), then_block
                ),
                bytecode.JmpInst(else_block)
            ]
        )

        self.assertEqual(entry_block, self.expr_emitter.parent_block)
        self.assertEqual(outer_end_block, self.expr_emitter.end_block)

        self.assertEqual(outer_result_var, self.expr_emitter.result)

    def test_conditional_in_conditional_test_expr(self) -> None:
        prog = sexp.parse('(if (if true false true) 42 43)')
        self.expr_emitter.visit(prog)

        end_block = bytecode.BasicBlock('bb6')

        body_result = bytecode.Var('var1')

        body_then_block = bytecode.BasicBlock(
            'bb4',
            [
                bytecode.CopyInst(
                    body_result, bytecode.NumLit(sexp.SNum(42))
                ),
                bytecode.JmpInst(end_block)
            ]
        )

        body_else_block = bytecode.BasicBlock(
            'bb5',
            [
                bytecode.CopyInst(
                    body_result, bytecode.NumLit(sexp.SNum(43))
                ),
                bytecode.JmpInst(end_block)
            ]
        )

        condition_result = bytecode.Var('var0')
        condition_end_block = bytecode.BasicBlock(
            'bb3',
            [
                bytecode.BrInst(condition_result, body_then_block),
                bytecode.JmpInst(body_else_block)
            ]
        )

        condition_then_block = bytecode.BasicBlock(
            'bb1',
            [
                bytecode.CopyInst(
                    condition_result, bytecode.BoolLit(sexp.SBool(False))
                ),
                bytecode.JmpInst(condition_end_block)
            ]
        )

        condition_else_block = bytecode.BasicBlock(
            'bb2',
            [
                bytecode.CopyInst(
                    condition_result, bytecode.BoolLit(sexp.SBool(True))
                ),
                bytecode.JmpInst(condition_end_block)
            ]
        )

        entry_block = bytecode.BasicBlock(
            'bb0',
            [
                bytecode.BrInst(
                    bytecode.BoolLit(sexp.SBool(True)), condition_then_block
                ),
                bytecode.JmpInst(condition_else_block)
            ]
        )

        self.assertEqual(entry_block, self.expr_emitter.parent_block)
        self.assertEqual(end_block, self.expr_emitter.end_block)

        self.assertEqual(body_result, self.expr_emitter.result)

    def test_conditional_in_function_call_args(self) -> None:
        prog = sexp.parse('(number? (if true 42 false))')
        self.expr_emitter.visit(prog)

        end_block = bytecode.BasicBlock('bb3')
        conditional_result = bytecode.Var('var1')

        then_block = bytecode.BasicBlock(
            'bb1',
            [
                bytecode.CopyInst(
                    conditional_result, bytecode.NumLit(sexp.SNum(42))
                ),
                bytecode.JmpInst(end_block)
            ]
        )

        else_block = bytecode.BasicBlock(
            'bb2',
            [
                bytecode.CopyInst(
                    conditional_result, bytecode.BoolLit(sexp.SBool(False))
                ),
                bytecode.JmpInst(end_block)
            ]
        )

        func_var = bytecode.Var('var0')
        typeof_var = bytecode.Var('__typeof')
        is_function_var = bytecode.Var('__is_func')
        entry_block = bytecode.BasicBlock(
            'bb0',
            [
                bytecode.LookupInst(
                    func_var, bytecode.SymLit(sexp.SSym('number?'))
                ),
                bytecode.TypeofInst(typeof_var, func_var),
                bytecode.BinopInst(
                    is_function_var, bytecode.Binop.SYM_EQ,
                    typeof_var, bytecode.SymLit(sexp.SSym('function'))
                ),
                bytecode.BrnInst(
                    is_function_var,
                    bytecode.BasicBlock(
                        '__non_function_trap',
                        [bytecode.TrapInst(
                            'Attempted to call a non-function')])
                ),

                bytecode.BrInst(
                    bytecode.BoolLit(sexp.SBool(True)), then_block
                ),
                bytecode.JmpInst(else_block)
            ]
        )

        call_result = bytecode.Var('var2')
        end_block.add_inst(
            bytecode.CallInst(call_result, func_var, [conditional_result]))

        self.assertEqual(entry_block, self.expr_emitter.parent_block)
        self.assertEqual(end_block, self.expr_emitter.end_block)

        self.assertEqual(call_result, self.expr_emitter.result)

    def test_conditional_in_function_call_func(self) -> None:
        prog = sexp.parse('((if true func1 func2))')
        self.expr_emitter.local_env[sexp.SSym('func1')] = (
            bytecode.Var('spam'))
        self.expr_emitter.local_env[sexp.SSym('func2')] = (
            bytecode.Var('egg'))
        self.expr_emitter.visit(prog)

        end_block = bytecode.BasicBlock('bb3')
        conditional_result = bytecode.Var('var0')

        then_block = bytecode.BasicBlock(
            'bb1',
            [
                bytecode.CopyInst(
                    conditional_result, bytecode.Var('spam')
                ),
                bytecode.JmpInst(end_block)
            ]
        )

        else_block = bytecode.BasicBlock(
            'bb2',
            [
                bytecode.CopyInst(
                    conditional_result, bytecode.Var('egg')
                ),
                bytecode.JmpInst(end_block)
            ]
        )

        entry_block = bytecode.BasicBlock(
            'bb0',
            [
                bytecode.BrInst(
                    bytecode.BoolLit(sexp.SBool(True)), then_block
                ),
                bytecode.JmpInst(else_block)
            ]
        )

        call_result = bytecode.Var('var1')
        typeof_var = bytecode.Var('__typeof')
        is_function_var = bytecode.Var('__is_func')
        call_instrs = [
            bytecode.TypeofInst(typeof_var, conditional_result),
            bytecode.BinopInst(
                is_function_var, bytecode.Binop.SYM_EQ,
                typeof_var, bytecode.SymLit(sexp.SSym('function'))
            ),
            bytecode.BrnInst(
                is_function_var,
                bytecode.BasicBlock(
                    '__non_function_trap',
                    [bytecode.TrapInst('Attempted to call a non-function')])
            ),

            bytecode.CallInst(call_result, conditional_result, [])
        ]

        for instr in call_instrs:
            end_block.add_inst(instr)

        self.assertEqual(entry_block, self.expr_emitter.parent_block)
        self.assertEqual(end_block, self.expr_emitter.end_block)

        self.assertEqual(call_result, self.expr_emitter.result)


class EmitFunctionDefTestCase(unittest.TestCase):
    function_emitter: emit_IR.FunctionEmitter

    def setUp(self) -> None:
        self.function_emitter = emit_IR.FunctionEmitter({})

    def test_emit_function_def(self) -> None:
        prog = sexp.parse('(define (func spam) true spam)')
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
        actual_func = self.function_emitter.global_env[sexp.SSym('func')]
        assert isinstance(actual_func, sexp.SFunction)
        self.assertEqual(expected, actual_func.code)

    def test_emit_multiple_function_defs(self) -> None:
        prog = sexp.parse('(define (func) 42) (define (func2) 43)')
        self.function_emitter.visit(prog)

        expected_func = bytecode.Function(
            [],
            bytecode.BasicBlock(
                'bb0',
                [bytecode.ReturnInst(bytecode.NumLit(sexp.SNum(42)))]
            )
        )

        expected_func2 = bytecode.Function(
            [],
            bytecode.BasicBlock(
                'bb0',
                [bytecode.ReturnInst(bytecode.NumLit(sexp.SNum(43)))]
            )
        )

        self.assertEqual(2, len(self.function_emitter.global_env))

        actual_func = self.function_emitter.global_env[sexp.SSym('func')]
        assert isinstance(actual_func, sexp.SFunction)
        self.assertEqual(expected_func, actual_func.code)

        actual_func2 = self.function_emitter.global_env[sexp.SSym('func2')]
        assert isinstance(actual_func2, sexp.SFunction)
        self.assertEqual(expected_func2, actual_func2.code)

    def test_emit_lambda_def(self) -> None:
        prog = sexp.parse('(define (func) (lambda (spam) spam))')
        self.function_emitter.visit(prog)

        func_ret_var = bytecode.Var('var0')
        expected_func = bytecode.Function(
            [],
            bytecode.BasicBlock(
                'bb0',
                [
                    bytecode.LookupInst(
                        func_ret_var, bytecode.SymLit(sexp.SSym('__lambda0'))
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

        actual_func = self.function_emitter.global_env[sexp.SSym('func')]
        assert isinstance(actual_func, sexp.SFunction)
        self.assertEqual(expected_func, actual_func.code)

        actual_lambda = (
            self.function_emitter.global_env[sexp.SSym('__lambda0')])
        assert isinstance(actual_lambda, sexp.SFunction)
        self.assertEqual(expected_lambda, actual_lambda.code)
