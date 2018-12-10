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
        self.assertEqual([scheme_types.SchemeSym], types)

    def test_quoted_list(self) -> None:
        prog = sexp.parse("'(1 2 3)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeVectType(2)], types)

    def test_vector_literal(self) -> None:
        prog = sexp.parse("[1 2 3 4]")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeVectType(4)], types)

    def test_vector_literal_size_above_specialization_threshold(self) -> None:
        prog = sexp.parse("[1 2 3 4 5]")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeVectType(None)], types)

    def test_num_literal(self) -> None:
        prog = sexp.parse("42")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeNum], types)

    def test_bool_literal(self) -> None:
        prog = sexp.parse("true false booly")
        analyzer = FunctionTypeAnalyzer(
            {sexp.SSym('booly'): scheme_types.SchemeBool}, {}
        )
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeBool,
            scheme_types.SchemeBool,
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
            scheme_types.SchemeNum,
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
            scheme_types.SchemeNum,

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
            scheme_types.SchemeBool,
            scheme_types.SchemeBool,
            scheme_types.SchemeBool,
        ]
        self.assertEqual(expected, types)

    def test_conditional_different_type_branches(self) -> None:
        prog = sexp.parse("(if spam 42 false)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeObject,
            scheme_types.SchemeNum,
            scheme_types.SchemeBool,
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
            scheme_types.SchemeNum,
            scheme_types.SchemeNum,
            scheme_types.SchemeNum,
        ]
        self.assertEqual(expected, types)

    def test_conditional_inexact_num_type_branches(self) -> None:
        prog = sexp.parse("(if spam 42 43)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeObject,
            scheme_types.SchemeNum,
            scheme_types.SchemeNum,
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

    def test_conditional_test_value_known_then_type_used(self) -> None:
        prog = sexp.parse("(if (number? 43) 42 false)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeFunctionType(
                arity=1, return_type=scheme_types.SchemeBool),
            scheme_types.SchemeNum,
            scheme_types.SchemeBool,

            scheme_types.SchemeNum,
            scheme_types.SchemeBool,
            scheme_types.SchemeNum,
        ]
        self.assertEqual(expected, types)

    def test_conditional_test_value_known_else_type_used(self) -> None:
        prog = sexp.parse("(if (number? true) 42 false)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            # test
            scheme_types.SchemeFunctionType(
                arity=1, return_type=scheme_types.SchemeBool),
            scheme_types.SchemeBool,
            scheme_types.SchemeBool,

            # then
            scheme_types.SchemeNum,
            # else
            scheme_types.SchemeBool,
            # overall
            scheme_types.SchemeBool,
        ]
        self.assertEqual(expected, types)

    def test_conditional_test_value_unknown_type_query_given_obj(self) -> None:
        prog = sexp.parse("(if (number? obj) 42 false)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeFunctionType(
                arity=1, return_type=scheme_types.SchemeBool),
            scheme_types.SchemeObject,
            scheme_types.SchemeBool,

            scheme_types.SchemeNum,
            scheme_types.SchemeBool,
            scheme_types.SchemeObject,
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
            scheme_types.SchemeNum,
            scheme_types.SchemeBool,
            scheme_types.SchemeVectType(3),
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
            scheme_types.SchemeBool,

            # return type
            scheme_types.SchemeVectType(None),

            # function type
            scheme_types.SchemeFunctionType(
                1, scheme_types.SchemeVectType(None)),
        ]
        self.assertEqual(expected, types)

    def test_vector_make_size_above_specialization_threshold(self) -> None:
        prog = sexp.parse("(vector-make 5 true)")
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeFunctionType(
                2, scheme_types.SchemeVectType(None)),
            scheme_types.SchemeNum,
            scheme_types.SchemeBool,
            scheme_types.SchemeVectType(None),
        ]
        self.assertEqual(expected, types)


class ConstExprTestCase(unittest.TestCase):
    def test_typeof_object(self) -> None:
        prog = sexp.parse('spam')
        analyzer = FunctionTypeAnalyzer({}, {})
        analyzer.visit(prog)

        values = list(analyzer.get_expr_values().values())
        self.assertEqual([], values)

    def test_typeof_value(self) -> None:
        prog = sexp.parse('''
            (typeof num)
            (typeof bool)
            (typeof sym)
            (typeof vec)
            (typeof func)
        ''')
        analyzer = FunctionTypeAnalyzer({
            sexp.SSym('num'): scheme_types.SchemeNum,
            sexp.SSym('bool'): scheme_types.SchemeBool,
            sexp.SSym('sym'): scheme_types.SchemeSym,
            sexp.SSym('vec'): scheme_types.SchemeVectType(2),
            sexp.SSym('func'): scheme_types.SchemeFunctionType(1),
        }, {})
        analyzer.visit(prog)

        values = list(analyzer.get_expr_values().values())
        expected = [
            sexp.SSym('number'),
            sexp.SSym('bool'),
            sexp.SSym('symbol'),
            sexp.SSym('vector'),
            sexp.SSym('function'),
        ]
        self.assertEqual(expected, values)
