import unittest

import bytecode
import runner
import scheme_types
import sexp
from scheme_types import FunctionTypeAnalyzer


class FunctionTypeAnalyzerTestCase(unittest.TestCase):
    def test_quoted_symbol(self) -> None:
        prog = sexp.parse("'spam")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeSymType('spam')], types)

    def test_quoted_list(self) -> None:
        prog = sexp.parse("'(1 2 3)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeVectType(2)], types)

    def test_num_literal(self) -> None:
        prog = sexp.parse("42")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeNumType(42)], types)

    def test_bool_literal(self) -> None:
        prog = sexp.parse("true false booly")
        analyzer = FunctionTypeAnalyzer(
            {sexp.SSym('booly'): scheme_types.SchemeBool}, {}
        )
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeBoolType(True),
            scheme_types.SchemeBoolType(False),
            scheme_types.SchemeBool
        ]
        self.assertEqual(expected, types)

    def test_sym_literal_not_function(self) -> None:
        prog = sexp.parse("spam")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeObject], types)

    def test_sym_literal_is_local_function(self) -> None:
        param_types = {sexp.SSym('spam'): scheme_types.SchemeFunctionType(1)}

        prog = sexp.parse("spam")
        analyzer = FunctionTypeAnalyzer(param_types, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeFunctionType(1)], types)

    def test_sym_literal_is_builtin_function(self) -> None:
        prog = sexp.parse("number=")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual(
            [scheme_types.SchemeFunctionType(2, scheme_types.SchemeBool)],
            types)

    def test_sym_literal_is_global_user_function(self) -> None:
        prog = sexp.parse("user_func")
        user_func = sexp.SFunction(
            sexp.SSym('user_func'),
            [sexp.SSym('param1'), sexp.SSym('param2'), sexp.SSym('param3')],
            sexp.to_slist([sexp.SBool(True)])
        )
        analyzer = FunctionTypeAnalyzer(
            {}, {sexp.SSym('user_func'): user_func})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual(
            [scheme_types.SchemeFunctionType(3, scheme_types.SchemeObject)],
            types)

    def test_function_def(self) -> None:
        prog = sexp.parse("(define (spam egg) egg)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())

        func_type = scheme_types.SchemeFunctionType(
            1, scheme_types.SchemeObject)
        expected = [
            scheme_types.SchemeObject,  # egg param symbol
            scheme_types.SchemeObject,  # egg usage symbol
            func_type,
        ]
        self.assertEqual(expected, types)
        self.assertEqual(func_type, analyzer.get_function_type())

    def test_function_call_type_unknown(self) -> None:
        prog = sexp.parse("(spam)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual(
            [scheme_types.SchemeObject, scheme_types.SchemeObject], types)

    def test_builtin_function_call_type(self) -> None:
        prog = sexp.parse("(number? 42)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeFunctionType(1, scheme_types.SchemeBool),
            scheme_types.SchemeNumType(42),
            scheme_types.SchemeBool
        ]
        self.assertEqual(expected, types)

    def test_user_function_return_type_deduced(self) -> None:
        prog = sexp.parse("(define (spam egg) (+ egg 1))")
        analyzer = FunctionTypeAnalyzer(
            {sexp.SSym('egg'): scheme_types.SchemeNum}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeNum,  # egg param

            # + signature
            scheme_types.SchemeFunctionType(2, scheme_types.SchemeNum),

            # Args to +
            scheme_types.SchemeNum,
            scheme_types.SchemeNumType(1),

            # + return val
            scheme_types.SchemeNum,

            # spam signature
            scheme_types.SchemeFunctionType(1, scheme_types.SchemeNum),
        ]
        self.assertEqual(expected, types)

    def test_conditional_same_type_branches(self) -> None:
        prog = sexp.parse("(if spam true false)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeObject,
            scheme_types.SchemeBoolType(True),
            scheme_types.SchemeBoolType(False),
            scheme_types.SchemeBool,
        ]
        self.assertEqual(expected, types)

    def test_conditional_test_true_type_is_then_branch(self) -> None:
        prog = sexp.parse("(if true true false)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeBoolType(True),
            scheme_types.SchemeBoolType(True),
            scheme_types.SchemeBoolType(False),
            scheme_types.SchemeBoolType(True),
        ]
        self.assertEqual(expected, types)

    def test_conditional_test_false_type_is_else_branch(self) -> None:
        prog = sexp.parse("(if false true false)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeBoolType(False),
            scheme_types.SchemeBoolType(True),
            scheme_types.SchemeBoolType(False),
            scheme_types.SchemeBoolType(False),
        ]
        self.assertEqual(expected, types)

    def test_conditional_different_type_branches(self) -> None:
        prog = sexp.parse("(if spam 42 false)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeObject,
            scheme_types.SchemeNumType(42),
            scheme_types.SchemeBoolType(False),
            scheme_types.SchemeObject,
        ]
        self.assertEqual(expected, types)

    def test_conditional_exact_num_type_branches(self) -> None:
        prog = sexp.parse("(if spam 42 42)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeObject,
            scheme_types.SchemeNumType(42),
            scheme_types.SchemeNumType(42),
            scheme_types.SchemeNumType(42),
        ]
        self.assertEqual(expected, types)

    def test_conditional_inexact_num_type_branches(self) -> None:
        prog = sexp.parse("(if spam 42 43)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeObject,
            scheme_types.SchemeNumType(42),
            scheme_types.SchemeNumType(43),
            scheme_types.SchemeNum,
        ]
        self.assertEqual(expected, types)

    def test_conditional_exact_func_type_branches(self) -> None:
        env = bytecode.EvalEnv()
        runner.add_builtins(env)
        prog = sexp.parse("(if spam number? number?)")
        analyzer = FunctionTypeAnalyzer({}, env._global_env)
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeObject,
            scheme_types.SchemeFunctionType(1, scheme_types.SchemeBool),
            scheme_types.SchemeFunctionType(1, scheme_types.SchemeBool),
            scheme_types.SchemeFunctionType(1, scheme_types.SchemeBool),
        ]
        self.assertEqual(expected, types)

    def test_conditional_inexact_func_type_branches(self) -> None:
        env = bytecode.EvalEnv()
        runner.add_builtins(env)
        prog = sexp.parse("(if spam number? number=)")
        analyzer = FunctionTypeAnalyzer({}, env._global_env)
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeObject,
            scheme_types.SchemeFunctionType(1, scheme_types.SchemeBool),
            scheme_types.SchemeFunctionType(2, scheme_types.SchemeBool),
            scheme_types.SchemeFunctionType(None, scheme_types.SchemeBool),
        ]
        self.assertEqual(expected, types)

    def test_conditional_inexact_func_return_type_branches(self) -> None:
        env = bytecode.EvalEnv()
        runner.add_builtins(env)
        prog = sexp.parse("(if egg number< +)")
        analyzer = FunctionTypeAnalyzer({}, env._global_env)
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeObject,
            scheme_types.SchemeFunctionType(2, scheme_types.SchemeBool),
            scheme_types.SchemeFunctionType(2, scheme_types.SchemeNum),
            scheme_types.SchemeFunctionType(2),
        ]
        self.assertEqual(expected, types)

    def test_conditional_exact_vect_type_branches(self) -> None:
        prog = sexp.parse("(if spam [1 2 3] [4 5 6])")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeObject,
            scheme_types.SchemeVectType(3),
            scheme_types.SchemeVectType(3),
            scheme_types.SchemeVectType(3),
        ]
        self.assertEqual(expected, types)

    def test_conditional_inexact_vect_type_branches(self) -> None:
        prog = sexp.parse("(if spam [1 2 3] [4 5])")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeObject,
            scheme_types.SchemeVectType(3),
            scheme_types.SchemeVectType(2),
            scheme_types.SchemeVectType(None),
        ]
        self.assertEqual(expected, types)

    def test_vector_make_literal_size_val(self) -> None:
        prog = sexp.parse("(vector-make 3 true)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeFunctionType(
                2, scheme_types.SchemeVectType(None)),
            scheme_types.SchemeNumType(3),
            scheme_types.SchemeBoolType(True),
            scheme_types.SchemeVectType(3),
        ]
        self.assertEqual(expected, types)

    def test_vector_make_known_size_param(self) -> None:
        prog = sexp.parse("(define (spam size) (vector-make size true))")
        analyzer = FunctionTypeAnalyzer(
            {sexp.SSym('size'): scheme_types.SchemeNumType(4)}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            # size param
            scheme_types.SchemeNumType(4),

            # vector-make
            scheme_types.SchemeFunctionType(
                2, scheme_types.SchemeVectType(None)),

            # size use
            scheme_types.SchemeNumType(4),
            scheme_types.SchemeBoolType(True),

            # return type
            scheme_types.SchemeVectType(4),

            # function type
            scheme_types.SchemeFunctionType(
                1, scheme_types.SchemeVectType(4)),
        ]
        self.assertEqual(expected, types)

    def test_vector_make_unknown_size_val(self) -> None:
        prog = sexp.parse("(define (spam size) (vector-make size true))")
        analyzer = FunctionTypeAnalyzer(
            {sexp.SSym('size'): scheme_types.SchemeNum}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            # size param
            scheme_types.SchemeNum,

            # vector-make
            scheme_types.SchemeFunctionType(
                2, scheme_types.SchemeVectType(None)),

            # size use
            scheme_types.SchemeNum,
            scheme_types.SchemeBoolType(True),

            # return type
            scheme_types.SchemeVectType(None),

            # function type
            scheme_types.SchemeFunctionType(
                1, scheme_types.SchemeVectType(None)),
        ]
        self.assertEqual(expected, types)
