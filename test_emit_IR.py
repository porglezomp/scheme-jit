import unittest
from typing import Dict, Optional

import bytecode
import emit_IR
import find_tail_calls
import runner
import scheme_types
import sexp


class EmitExpressionTestCase(unittest.TestCase):
    expr_emitter: emit_IR.ExpressionEmitter
    bb: bytecode.BasicBlock

    def setUp(self) -> None:
        bb_names = emit_IR.name_generator('bb')
        self.bb = bytecode.BasicBlock(next(bb_names))
        self.expr_emitter = emit_IR.ExpressionEmitter(
            self.bb, bb_names, emit_IR.name_generator('v'), {}, {})

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
                bytecode.Var('v0'),
                bytecode.NumLit(sexp.SNum(0))
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_non_empty_vector_literal(self) -> None:
        prog = sexp.parse('[1 2]')
        self.expr_emitter.visit(prog)

        arr_var = bytecode.Var('v0')
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

        nil_var = bytecode.Var('v0')
        expected_instrs = [
            bytecode.AllocInst(
                nil_var, bytecode.NumLit(sexp.SNum(0))
            )
        ]

        self.assertEqual(expected_instrs, self.bb.instructions)

    def test_emit_non_empty_quoted_list(self) -> None:
        prog = sexp.parse("'(1 spam)")
        self.expr_emitter.visit(prog)

        nil_var = bytecode.Var('v0')
        second_pair = bytecode.Var('v1')
        first_pair = bytecode.Var('v2')
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

        func_var = bytecode.Var('v0')
        typeof_var = bytecode.Var('v1')
        is_function_var = bytecode.Var('v2')
        arity_var = bytecode.Var('v3')
        correct_arity_var = bytecode.Var('v4')
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
                    'non_function',
                    [bytecode.TrapInst('Attempted to call a non-function')])
            ),

            bytecode.ArityInst(arity_var, func_var),
            bytecode.BinopInst(
                correct_arity_var, bytecode.Binop.NUM_EQ,
                arity_var, bytecode.NumLit(sexp.SNum(1))
            ),
            bytecode.BrnInst(
                correct_arity_var,
                bytecode.BasicBlock(
                    'wrong_arity',
                    [bytecode.TrapInst(
                        'Call with the wrong number of arguments')])
            ),

            bytecode.CallInst(
                bytecode.Var('v5'),
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

        typeof_var = bytecode.Var('v0')
        is_function_var = bytecode.Var('v1')
        arity_var = bytecode.Var('v2')
        correct_arity_var = bytecode.Var('v3')
        expected_instrs = [
            bytecode.TypeofInst(typeof_var, lambda_var),
            bytecode.BinopInst(
                is_function_var, bytecode.Binop.SYM_EQ,
                typeof_var, bytecode.SymLit(sexp.SSym('function'))
            ),
            bytecode.BrnInst(
                is_function_var,
                bytecode.BasicBlock(
                    'non_function',
                    [bytecode.TrapInst('Attempted to call a non-function')])
            ),

            bytecode.ArityInst(arity_var, lambda_var),
            bytecode.BinopInst(
                correct_arity_var, bytecode.Binop.NUM_EQ,
                arity_var, bytecode.NumLit(sexp.SNum(1))
            ),
            bytecode.BrnInst(
                correct_arity_var,
                bytecode.BasicBlock(
                    'wrong_arity',
                    [bytecode.TrapInst(
                        'Call with the wrong number of arguments')])
            ),

            bytecode.CallInst(
                bytecode.Var('v4'),
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
                'bb0', [bytecode.ReturnInst(bytecode.Var('spam'))]
            )
        )

        lambda_lookup_var = bytecode.Var('v0')
        typeof_var = bytecode.Var('v1')
        is_function_var = bytecode.Var('v2')
        arity_var = bytecode.Var('v3')
        correct_arity_var = bytecode.Var('v4')
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
                    'non_function',
                    [bytecode.TrapInst('Attempted to call a non-function')])
            ),

            bytecode.ArityInst(arity_var, lambda_lookup_var),
            bytecode.BinopInst(
                correct_arity_var, bytecode.Binop.NUM_EQ,
                arity_var, bytecode.NumLit(sexp.SNum(1))
            ),
            bytecode.BrnInst(
                correct_arity_var,
                bytecode.BasicBlock(
                    'wrong_arity',
                    [bytecode.TrapInst(
                        'Call with the wrong number of arguments')])
            ),

            bytecode.CallInst(
                bytecode.Var('v5'),
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

        result_var = bytecode.Var('v0')
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

        outer_result_var = bytecode.Var('v0')

        outer_end_block = bytecode.BasicBlock('bb9')

        then_result_var = bytecode.Var('v1')
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

        else_result_var = bytecode.Var('v2')
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

        body_result = bytecode.Var('v1')

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

        condition_result = bytecode.Var('v0')
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
        func_var = bytecode.Var('v0')
        typeof_var = bytecode.Var('v1')
        is_function_var = bytecode.Var('v2')
        arity_var = bytecode.Var('v3')
        correct_arity_var = bytecode.Var('v4')
        conditional_result = bytecode.Var('v5')
        call_result = bytecode.Var('v6')

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
                        'non_function',
                        [bytecode.TrapInst(
                            'Attempted to call a non-function')])
                ),

                bytecode.ArityInst(arity_var, func_var),
                bytecode.BinopInst(
                    correct_arity_var, bytecode.Binop.NUM_EQ,
                    arity_var, bytecode.NumLit(sexp.SNum(1))
                ),
                bytecode.BrnInst(
                    correct_arity_var,
                    bytecode.BasicBlock(
                        'wrong_arity',
                        [bytecode.TrapInst(
                            'Call with the wrong number of arguments')])
                ),

                bytecode.BrInst(
                    bytecode.BoolLit(sexp.SBool(True)), then_block
                ),
                bytecode.JmpInst(else_block)
            ]
        )

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
        conditional_result = bytecode.Var('v0')
        typeof_var = bytecode.Var('v1')
        is_function_var = bytecode.Var('v2')
        arity_var = bytecode.Var('v3')
        correct_arity_var = bytecode.Var('v4')
        call_result = bytecode.Var('v5')

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

        call_instrs = [
            bytecode.TypeofInst(typeof_var, conditional_result),
            bytecode.BinopInst(
                is_function_var, bytecode.Binop.SYM_EQ,
                typeof_var, bytecode.SymLit(sexp.SSym('function'))
            ),
            bytecode.BrnInst(
                is_function_var,
                bytecode.BasicBlock(
                    'non_function',
                    [bytecode.TrapInst('Attempted to call a non-function')])
            ),

            bytecode.ArityInst(arity_var, conditional_result),
            bytecode.BinopInst(
                correct_arity_var, bytecode.Binop.NUM_EQ,
                arity_var, bytecode.NumLit(sexp.SNum(0))
            ),
            bytecode.BrnInst(
                correct_arity_var,
                bytecode.BasicBlock(
                    'wrong_arity',
                    [bytecode.TrapInst(
                        'Call with the wrong number of arguments')])
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
                'bb0', [bytecode.ReturnInst(return_var)]
            )
        )

        self.assertEqual(expected, self.function_emitter.get_emitted_func())

    def test_emit_multiple_function_defs(self) -> None:
        prog = sexp.parse('(define (func) 42) (define (func2) 43)')

        self.function_emitter.visit(prog[0])

        expected_func = bytecode.Function(
            [],
            bytecode.BasicBlock(
                'bb0',
                [bytecode.ReturnInst(bytecode.NumLit(sexp.SNum(42)))]
            )
        )
        self.assertEqual(
            expected_func, self.function_emitter.get_emitted_func())

        self.function_emitter.visit(prog[1])
        expected_func2 = bytecode.Function(
            [],
            bytecode.BasicBlock(
                'bb0',
                [bytecode.ReturnInst(bytecode.NumLit(sexp.SNum(43)))]
            )
        )
        self.assertEqual(
            expected_func2, self.function_emitter.get_emitted_func())

    def test_emit_lambda_def(self) -> None:
        prog = sexp.parse('(define (func) (lambda (spam) spam))')
        self.function_emitter.visit(prog)

        func_ret_var = bytecode.Var('v0')
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
               'bb0', [bytecode.ReturnInst(bytecode.Var('spam'))]
            )
        )

        self.assertEqual(expected_func,
                         self.function_emitter.get_emitted_func())

        actual_lambda = (
            self.function_emitter.global_env[sexp.SSym('__lambda0')])
        assert isinstance(actual_lambda, sexp.SFunction)
        self.assertEqual(expected_lambda, actual_lambda.code)

    def test_emit_begin_in_func(self) -> None:
        self.maxDiff = None
        prog = sexp.parse('(define (spam) (begin (trace 42) (trace 43) 44))')
        self.function_emitter.visit(prog)

        expected = '''
function (?) entry=bb0
bb0:
  v0 = lookup 'trace
  v1 = typeof v0
  v2 = Binop.SYM_EQ v1 'function
  brn v2 non_function
  v3 = arity v0
  v4 = Binop.NUM_EQ v3 1
  brn v4 wrong_arity
  v5 = call v0 (42)
  v6 = lookup 'trace
  v7 = typeof v6
  v8 = Binop.SYM_EQ v7 'function
  brn v8 non_function
  v9 = arity v6
  v10 = Binop.NUM_EQ v9 1
  brn v10 wrong_arity
  v11 = call v6 (43)
  return 44

non_function:
  trap 'Attempted to call a non-function'

wrong_arity:
  trap 'Call with the wrong number of arguments'
        '''

        self.assertEqual(
            expected.strip(),
            str(self.function_emitter.get_emitted_func()).strip())


class EmitOptimizedFuncTestCase(unittest.TestCase):
    env: bytecode.EvalEnv

    def setUp(self) -> None:
        self.env = bytecode.EvalEnv()
        runner.add_intrinsics(self.env)
        runner.add_builtins(self.env)
        runner.add_prelude(self.env)

    def get_optimized_func_bytecode(
            self, code_str: str,
            param_types: Optional[Dict[sexp.SSym,
                                       scheme_types.SchemeObjectType]] = None,
            optimize_tail_calls: bool = False) -> str:
        [func] = sexp.parse(code_str)
        assert isinstance(func, sexp.SFunction)

        tail_calls = None
        if optimize_tail_calls:
            tail_call_finder = find_tail_calls.TailCallFinder()
            tail_call_finder.visit(func)
            tail_calls = tail_call_finder.tail_calls

        # Make sure func is in global env
        emitter = emit_IR.FunctionEmitter(self.env._global_env)
        emitter.visit(func)
        func.code = emitter.get_emitted_func()
        self.env._global_env[func.name] = func

        type_analyzer = None
        if param_types is not None:
            type_analyzer = scheme_types.FunctionTypeAnalyzer(
                param_types=param_types,
                global_env=self.env._global_env)
            type_analyzer.visit(func)

        optimizing_emitter = emit_IR.FunctionEmitter(
            self.env._global_env,
            tail_calls=tail_calls,
            expr_types=type_analyzer)
        optimizing_emitter.visit(func)

        return str(optimizing_emitter.get_emitted_func())

    # -------------------------------------------------------------------------

    def test_partially_specialized_plus(self) -> None:
        code = '''
            (define (plus first second)
                (assert (number? first))
                (assert (number? second))
                (inst/+ first second)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('first'): scheme_types.SchemeNum,
                sexp.SSym('second'): scheme_types.SchemeObject
            }
        )

        expected = '''
function (? first second) entry=bb0
bb0:
  v0 = lookup 'assert
  v1 = lookup 'number?
  v2 = call v1 (second) (object)
  v3 = call v0 (v2) (bool)
  v4 = lookup 'inst/+
  v5 = call v4 (first, second) (number, object)
  return v5
'''

        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_fully_specialized_plus(self) -> None:
        code = '''
            (define (plus first second)
                (assert (number? first))
                (assert (number? second))
                (inst/+ first second)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('first'): scheme_types.SchemeNum,
                sexp.SSym('second'): scheme_types.SchemeNum
            }
        )

        expected = '''
function (? first second) entry=bb0
bb0:
  v0 = lookup 'inst/+
  v1 = call v0 (first, second) (number, number)
  return v1
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_fully_specialized_plus_args_are_exprs(self) -> None:
        code = '''
            (define (plus first second)
                (assert (number? first))
                (assert (number? second))
                (inst/+ (+ 1 first) (+ second 1))
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('first'): scheme_types.SchemeNum,
                sexp.SSym('second'): scheme_types.SchemeNum
            }
        )

        expected = '''
function (? first second) entry=bb0
bb0:
  v0 = lookup 'inst/+
  v1 = lookup '+
  v2 = call v1 (1, first) (number, number)
  v3 = lookup '+
  v4 = call v3 (second, 1) (number, number)
  v5 = call v0 (v2, v4) (number, number)
  return v5
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_specialize_in_bounds_vector_access(self) -> None:
        code = '''
            (define (pairy pair)
                (vector-index pair 0)
                (vector-set! pair 1 42)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('pair'): scheme_types.SchemeVectType(2),
            }
        )

        expected = '''
function (? pair) entry=bb0
bb0:
  v0 = lookup 'inst/load
  v1 = call v0 (pair, 0) (vector[2], number)
  v2 = lookup 'inst/store
  v3 = call v2 (pair, 1, 42) (vector[2], number, number)
  return v3
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_out_of_bounds_vector_access_checks_not_removed(self) -> None:
        code = '''
            (define (pairy pair)
                (vector-index pair 2)
                (vector-set! pair 2 42)

                (vector-index pair -1)
                (vector-set! pair -1 42)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('pair'): scheme_types.SchemeVectType(2),
            }
        )

        expected = '''
function (? pair) entry=bb0
bb0:
  v0 = lookup 'vector-index
  v1 = call v0 (pair, 2) (vector[2], number)
  v2 = lookup 'vector-set!
  v3 = call v2 (pair, 2, 42) (vector[2], number, number)
  v4 = lookup 'vector-index
  v5 = call v4 (pair, -1) (vector[2], number)
  v6 = lookup 'vector-set!
  v7 = call v6 (pair, -1, 42) (vector[2], number, number)
  return v7
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_non_constant_index_checks_not_removed(self) -> None:
        code = '''
            (define (pairy pair)
                (vector-index pair index)
                (vector-set! pair index 42)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('pair'): scheme_types.SchemeVectType(2),
            }
        )

        expected = '''
function (? pair) entry=bb0
bb0:
  v0 = lookup 'vector-index
  v1 = lookup 'index
  v2 = call v0 (pair, v1) (vector[2], object)
  v3 = lookup 'vector-set!
  v4 = lookup 'index
  v5 = call v3 (pair, v4, 42) (vector[2], object, number)
  return v5
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_vector_access_unknown_vector_size(self) -> None:
        code = '''
            (define (pairy pair)
                (vector-index pair 0)
                (vector-set! pair 1 42)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('pair'): scheme_types.SchemeVectType(None),
            }
        )

        expected = '''
function (? pair) entry=bb0
bb0:
  v0 = lookup 'vector-index
  v1 = call v0 (pair, 0) (vector, number)
  v2 = lookup 'vector-set!
  v3 = call v2 (pair, 1, 42) (vector, number, number)
  return v3
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_vector_access_non_vector(self) -> None:
        code = '''
            (define (pairy pair)
                (vector-index pair 0)
                (vector-set! pair 1 42)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('pair'): scheme_types.SchemeObject,
            }
        )

        expected = '''
function (? pair) entry=bb0
bb0:
  v0 = lookup 'vector-index
  v1 = call v0 (pair, 0) (object, number)
  v2 = lookup 'vector-set!
  v3 = call v2 (pair, 1, 42) (object, number, number)
  return v3
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_vector_access_non_number_index(self) -> None:
        code = '''
            (define (pairy pair)
                (vector-index pair true)
                (vector-set! pair false 42)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('pair'): scheme_types.SchemeVectType(2),
            }
        )

        expected = '''
function (? pair) entry=bb0
bb0:
  v0 = lookup 'vector-index
  v1 = call v0 (pair, True) (vector[2], bool)
  v2 = lookup 'vector-set!
  v3 = call v2 (pair, False, 42) (vector[2], bool, number)
  return v3
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_removed_is_func_and_arity_check_param_is_func(self) -> None:
        code = '''
            (define (spam func)
                (func 42)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('func'): scheme_types.SchemeFunctionType(1)
            }
        )

        expected = '''
function (? func) entry=bb0
bb0:
  v0 = call func (42) (number)
  return v0
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_removed_is_func_check_param_is_func_wrong_num_args(self) -> None:
        code = '''
            (define (spam func)
                (func 42 43)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('func'): scheme_types.SchemeFunctionType(1)
            }
        )

        expected = '''
function (? func) entry=bb0
bb0:
  v0 = arity func
  v1 = Binop.NUM_EQ v0 2
  brn v1 wrong_arity
  v2 = call func (42, 43) (number, number)
  return v2

wrong_arity:
  trap 'Call with the wrong number of arguments'
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_removed_is_func_check_param_is_func_arity_unknown(self) -> None:
        code = '''
            (define (spam func)
                (func 42)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('func'): scheme_types.SchemeFunctionType(None)
            }
        )

        expected = '''
function (? func) entry=bb0
bb0:
  v0 = arity func
  v1 = Binop.NUM_EQ v0 1
  brn v1 wrong_arity
  v2 = call func (42) (number)
  return v2

wrong_arity:
  trap 'Call with the wrong number of arguments'
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_removed_is_func_and_arity_check_lambda_call(self) -> None:
        code = '''
            (define (spam)
                ((lambda (egg) egg) 42)
            )'''
        optimized = self.get_optimized_func_bytecode(code, param_types={})

        expected = '''
function (?) entry=bb0
bb0:
  v0 = lookup '__lambda0
  v1 = call v0 (42) (number)
  return v1
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_removed_is_func_check_lambda_call(self) -> None:
        code = '''
            (define (spam)
                ((lambda (egg) egg) 42 43)
            )'''
        optimized = self.get_optimized_func_bytecode(code, param_types={})

        expected = '''
function (?) entry=bb0
bb0:
  v0 = lookup '__lambda0
  v1 = arity v0
  v2 = Binop.NUM_EQ v1 2
  brn v2 wrong_arity
  v3 = call v0 (42, 43) (number, number)
  return v3

wrong_arity:
  trap 'Call with the wrong number of arguments'
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_removed_is_func_and_arity_check_user_func_call(self) -> None:
        code = '''(define (spam) (spam))'''
        optimized = self.get_optimized_func_bytecode(code, param_types={})

        expected = '''
function (?) entry=bb0
bb0:
  v0 = lookup 'spam
  v1 = call v0 ()
  return v1
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_removed_is_func_check_user_func_call(self) -> None:
        code = '''(define (spam) (spam 42))'''
        optimized = self.get_optimized_func_bytecode(code, param_types={})

        expected = '''
function (?) entry=bb0
bb0:
  v0 = lookup 'spam
  v1 = arity v0
  v2 = Binop.NUM_EQ v1 1
  brn v2 wrong_arity
  v3 = call v0 (42) (number)
  return v3

wrong_arity:
  trap 'Call with the wrong number of arguments'
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_removed_is_func_and_arity_check_builtin_func_call(self) -> None:
        code = '''(define (spam) (+ 3 4))'''
        optimized = self.get_optimized_func_bytecode(code, param_types={})

        expected = '''
function (?) entry=bb0
bb0:
  v0 = lookup '+
  v1 = call v0 (3, 4) (number, number)
  return v1
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_removed_is_func_check_builtin_call_with_type_info(self) -> None:
        code = '''(define (spam) (bool? 3 4))'''
        optimized = self.get_optimized_func_bytecode(code, param_types={})

        expected = '''
function (?) entry=bb0
bb0:
  v0 = lookup 'bool?
  v1 = arity v0
  v2 = Binop.NUM_EQ v1 2
  brn v2 wrong_arity
  v3 = call v0 (3, 4) (number, number)
  return v3

wrong_arity:
  trap 'Call with the wrong number of arguments'
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_would_be_tail_call_wrong_arity_with_type_info(self) -> None:
        code = '''(define (spam) (spam 42))'''
        optimized = self.get_optimized_func_bytecode(
            code, {}, optimize_tail_calls=True)

        expected = '''
function (?) entry=bb0
bb0:
  v0 = lookup 'spam
  v1 = arity v0
  v2 = Binop.NUM_EQ v1 1
  brn v2 wrong_arity
  v3 = call v0 (42) (number)
  return v3

wrong_arity:
  trap 'Call with the wrong number of arguments'
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_would_be_tail_call_wrong_arity_no_type_info(self) -> None:
        code = '''(define (spam) (spam 42))'''
        optimized = self.get_optimized_func_bytecode(
            code, optimize_tail_calls=True)

        expected = '''
function (?) entry=bb0
bb0:
  v0 = lookup 'spam
  v1 = typeof v0
  v2 = Binop.SYM_EQ v1 'function
  brn v2 non_function
  v3 = arity v0
  v4 = Binop.NUM_EQ v3 1
  brn v4 wrong_arity
  v5 = call v0 (42)
  return v5

non_function:
  trap 'Attempted to call a non-function'

wrong_arity:
  trap 'Call with the wrong number of arguments'
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_tail_call_right_arity_no_type_info(self) -> None:
        code = '''(define (spam egg) (spam 42))'''
        optimized = self.get_optimized_func_bytecode(
            code, optimize_tail_calls=True)

        expected = '''
function (? egg) entry=bb0
bb0:
  egg = 42
  jmp bb0
  return 0
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_tail_call_vector_make_recur_no_type_info(self) -> None:
        code = '''
            (define (vector-make/recur len idx v x)
                (vector-set! v idx x)
                (if (number= len (+ idx 1))
                    v
                    (vector-make/recur len (+ idx 1) v x)
                )
            )
            '''
        optimized = self.get_optimized_func_bytecode(
            code, optimize_tail_calls=True)

        expected = '''
function (? len idx v x) entry=bb0
bb0:
  v0 = lookup 'vector-set!
  v1 = typeof v0
  v2 = Binop.SYM_EQ v1 'function
  brn v2 non_function
  v3 = arity v0
  v4 = Binop.NUM_EQ v3 3
  brn v4 wrong_arity
  v5 = call v0 (v, idx, x)
  v6 = lookup 'number=
  v7 = typeof v6
  v8 = Binop.SYM_EQ v7 'function
  brn v8 non_function
  v9 = arity v6
  v10 = Binop.NUM_EQ v9 2
  brn v10 wrong_arity
  v11 = lookup '+
  v12 = typeof v11
  v13 = Binop.SYM_EQ v12 'function
  brn v13 non_function
  v14 = arity v11
  v15 = Binop.NUM_EQ v14 2
  brn v15 wrong_arity
  v16 = call v11 (idx, 1)
  v17 = call v6 (len, v16)
  br v17 bb1
  jmp bb2

non_function:
  trap 'Attempted to call a non-function'

wrong_arity:
  trap 'Call with the wrong number of arguments'

bb1:
  v18 = v
  jmp bb3

bb2:
  v19 = lookup '+
  v20 = typeof v19
  v21 = Binop.SYM_EQ v20 'function
  brn v21 non_function
  v22 = arity v19
  v23 = Binop.NUM_EQ v22 2
  brn v23 wrong_arity
  v24 = call v19 (idx, 1)
  idx = v24
  jmp bb0
  v18 = 0
  jmp bb3

bb3:
  return v18
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_tail_call_in_specialized(self) -> None:
        code = '''
            (define (spam egg)
                (assert (number? egg))
                (spam (+ egg 1))
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('egg'): scheme_types.SchemeNum
            },
            optimize_tail_calls=True)

        expected = '''
function (? egg) entry=bb0
bb0:
  v0 = lookup '+
  v1 = call v0 (egg, 1) (number, number)
  egg = v1
  jmp bb0
  return 0
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_tail_call_in_specialized_vector_make_recur(self) -> None:
        self.maxDiff = None
        code = '''
            (define (vector-make/recur len idx v x)
                (vector-set! v idx x)
                (if (number= len (+ idx 1))
                    v
                    (vector-make/recur len (+ idx 1) v x)
                )
            )
            '''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('len'): scheme_types.SchemeNum,
                sexp.SSym('idx'): scheme_types.SchemeNum,
                sexp.SSym('v'): scheme_types.SchemeVectType(None),
                sexp.SSym('x'): scheme_types.SchemeNum,
            },
            optimize_tail_calls=True)

        expected = '''
function (? len idx v x) entry=bb0
bb0:
  v0 = lookup 'vector-set!
  v1 = call v0 (v, idx, x) (vector, number, number)
  v2 = lookup 'number=
  v3 = lookup '+
  v4 = call v3 (idx, 1) (number, number)
  v5 = call v2 (len, v4) (number, number)
  br v5 bb1
  jmp bb2

bb1:
  v6 = v
  jmp bb3

bb2:
  v7 = lookup '+
  v8 = call v7 (idx, 1) (number, number)
  idx = v8
  jmp bb0
  v6 = 0
  jmp bb3

bb3:
  return v6
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_tail_call_in_specialized_code_mismatching_arg_types(self) -> None:
        code = '''
            (define (spam egg)
                (assert (number? egg))
                (spam true)
            )'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('egg'): scheme_types.SchemeNum
            },
            optimize_tail_calls=True)

        expected = '''
function (? egg) entry=bb0
bb0:
  v0 = lookup 'spam
  v1 = call v0 (True) (bool)
  return v1
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_remove_is_function_assertion_arity_known(self) -> None:
        code = '''(define (spam func) (assert (function? func)))'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('func'): scheme_types.SchemeFunctionType(1)
            }
        )

        expected = '''
function (? func) entry=bb0
bb0:
  return 0
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_remove_is_function_assertion_arity_unknown(self) -> None:
        code = '''(define (spam func) (assert (function? func)))'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('func'): scheme_types.SchemeFunctionType(None)
            }
        )

        expected = '''
function (? func) entry=bb0
bb0:
  return 0
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_non_func_assertion_not_removed(self) -> None:
        code = '''(define (spam func) (assert (function? func)))'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('func'): scheme_types.SchemeObject
            }
        )

        expected = '''
function (? func) entry=bb0
bb0:
  v0 = lookup 'assert
  v1 = lookup 'function?
  v2 = call v1 (func) (object)
  v3 = call v0 (v2) (bool)
  return v3
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_remove_is_vector_assertion_size_known(self) -> None:
        code = '''(define (spam vec) (assert (vector? vec)))'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('vec'): scheme_types.SchemeVectType(4)
            }
        )

        expected = '''
function (? vec) entry=bb0
bb0:
  return 0
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_remove_is_vector_assertion_size_unknown(self) -> None:
        code = '''(define (spam vec) (assert (vector? vec)))'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('vec'): scheme_types.SchemeVectType(None)
            }
        )

        expected = '''
function (? vec) entry=bb0
bb0:
  return 0
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_no_remove_pair_assertion(self) -> None:
        code = '''(define (spam vec) (assert (pair? vec)))'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('vec'): scheme_types.SchemeVectType(2)
            }
        )

        expected = '''
function (? vec) entry=bb0
bb0:
  v0 = lookup 'assert
  v1 = lookup 'pair?
  v2 = call v1 (vec) (vector[2])
  v3 = call v0 (v2) (bool)
  return v3
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_no_remove_nil_assertion(self) -> None:
        code = '''(define (spam vec) (assert (nil? vec)))'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('vec'): scheme_types.SchemeVectType(0)
            }
        )

        expected = '''
function (? vec) entry=bb0
bb0:
  v0 = lookup 'assert
  v1 = lookup 'nil?
  v2 = call v1 (vec) (vector[0])
  v3 = call v0 (v2) (bool)
  return v3
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_non_vec_assert_not_removed(self) -> None:
        code = '''(define (spam egg) (assert (vector? egg)))'''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('egg'): scheme_types.SchemeObject
            }
        )

        expected = '''
function (? egg) entry=bb0
bb0:
  v0 = lookup 'assert
  v1 = lookup 'vector?
  v2 = call v1 (egg) (object)
  v3 = call v0 (v2) (bool)
  return v3
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    # -------------------------------------------------------------------------

    def test_eq_specialization_both_num(self) -> None:
        code = '''
            (define (= x y)
                (if (not (symbol= (typeof x) (typeof y)))
                    false
                    (if (symbol? x)
                        (symbol= x y)
                        (if (number? x)
                            (number= x y)
                            (if (vector? x)
                                (vector= x y)
                                (pointer= x y))))))
        '''
        optimized = self.get_optimized_func_bytecode(
            code,
            param_types={
                sexp.SSym('x'): scheme_types.SchemeNum,
                sexp.SSym('y'): scheme_types.SchemeNum,
            }
        )

        expected = '''
function (? x y) entry=bb0
bb0:
  v0 = lookup 'number=
  v1 = call v0 (x, y) (number, number)
  return v1
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_eq_specialization_both_sym_lit(self) -> None:
        optimized = self.get_optimized_func_bytecode(
            _EQUAL_CODE,
            param_types={
                sexp.SSym('x'): scheme_types.SchemeSym,
                sexp.SSym('y'): scheme_types.SchemeSym,
            }
        )

        expected = '''
function (? x y) entry=bb0
bb0:
  v0 = lookup 'symbol=
  v1 = call v0 (x, y) (symbol, symbol)
  return v1
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_eq_specialization_both_vector(self) -> None:
        optimized = self.get_optimized_func_bytecode(
            _EQUAL_CODE,
            param_types={
                # Length shouldn't matter here
                sexp.SSym('x'): scheme_types.SchemeVectType(3),
                sexp.SSym('y'): scheme_types.SchemeVectType(1),
            }
        )

        expected = '''
function (? x y) entry=bb0
bb0:
  v0 = lookup 'vector=
  v1 = call v0 (x, y) (vector[3], vector[1])
  return v1
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_eq_specialization_both_bool(self) -> None:
        optimized = self.get_optimized_func_bytecode(
            _EQUAL_CODE,
            param_types={
                sexp.SSym('x'): scheme_types.SchemeBool,
                sexp.SSym('y'): scheme_types.SchemeBool,
            }
        )

        expected = '''
function (? x y) entry=bb0
bb0:
  v0 = lookup 'pointer=
  v1 = call v0 (x, y) (bool, bool)
  return v1
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_eq_specialization_both_func(self) -> None:
        optimized = self.get_optimized_func_bytecode(
            _EQUAL_CODE,
            param_types={
                # Arity shouldn't matter here
                sexp.SSym('x'): scheme_types.SchemeFunctionType(1),
                sexp.SSym('y'): scheme_types.SchemeFunctionType(2),
            }
        )

        expected = '''
function (? x y) entry=bb0
bb0:
  v0 = lookup 'pointer=
  v1 = call v0 (x, y) (function[1, object], function[2, object])
  return v1
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_eq_specialization_different_types(self) -> None:
        optimized = self.get_optimized_func_bytecode(
            _EQUAL_CODE,
            param_types={
                sexp.SSym('x'): scheme_types.SchemeBool,
                sexp.SSym('y'): scheme_types.SchemeSym,
            }
        )

        expected = '''
function (? x y) entry=bb0
bb0:
  return False
        '''
        self.assertEqual(expected.strip(), optimized.strip())

    def test_remove_branch(self) -> None:
        optimized = self.get_optimized_func_bytecode(
            '(define (f x) (if false 1 (if x 2 3)))',
            param_types={}
        )
        expected = '''
function (? x) entry=bb0
bb0:
  br x bb1
  jmp bb2

bb1:
  v0 = 2
  jmp bb3

bb2:
  v0 = 3
  jmp bb3

bb3:
  return v0
        '''
        self.assertEqual(expected.strip(), optimized.strip())


_EQUAL_CODE = '''
    (define (equal x y)
        (if (not (symbol= (typeof x) (typeof y)))
            false
            (if (symbol? x)
                (symbol= x y)
                (if (number? x)
                    (number= x y)
                    (if (vector? x)
                        (vector= x y)
                        (pointer= x y)
                    )
                )
            )
        )
    )'''
