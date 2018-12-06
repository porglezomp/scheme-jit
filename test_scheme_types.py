import unittest

import scheme_types
import sexp
from scheme_types import FunctionTypeAnalyzer


class FunctionTypeAnalyzerTestCase(unittest.TestCase):
    def test_quoted_symbol(self) -> None:
        prog = sexp.parse("'spam")
        analyzer = FunctionTypeAnalyzer({})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeQuotedType(sexp.SSym)], types)

    def test_quoted_list(self) -> None:
        prog = sexp.parse("'(1 2 3)")
        analyzer = FunctionTypeAnalyzer({})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeVectType(2)], types)

    def test_num_literal(self) -> None:
        prog = sexp.parse("42")
        analyzer = FunctionTypeAnalyzer({})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeNum], types)

    def test_bool_literal(self) -> None:
        prog = sexp.parse("true")
        analyzer = FunctionTypeAnalyzer({})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeBool], types)

    def test_sym_literal_not_function(self) -> None:
        prog = sexp.parse("spam")
        analyzer = FunctionTypeAnalyzer({})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeObject], types)

    def test_sym_literal_is_local_function(self) -> None:
        param_types = {sexp.SSym('spam'): scheme_types.SchemeFunctionType(1)}

        prog = sexp.parse("spam")
        analyzer = FunctionTypeAnalyzer(param_types)
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual([scheme_types.SchemeFunctionType(1)], types)

    def test_sym_literal_is_builtin_function(self) -> None:
        prog = sexp.parse("number=")
        analyzer = FunctionTypeAnalyzer({})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual(
            [scheme_types.SchemeFunctionType(2, scheme_types.SchemeBool)],
            types)

    def test_function_def(self) -> None:
        prog = sexp.parse("(define (spam egg) egg)")
        analyzer = FunctionTypeAnalyzer({})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())

        func_type = scheme_types.SchemeFunctionType(
            1, scheme_types.SchemeObject)
        expected = [
            scheme_types.SchemeObject,
            func_type,
        ]
        self.assertEqual(expected, types)
        self.assertEqual(func_type, analyzer.get_function_type())

    def test_function_call_type_unknown(self) -> None:
        prog = sexp.parse("(spam)")
        analyzer = FunctionTypeAnalyzer({})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        self.assertEqual(
            [scheme_types.SchemeObject, scheme_types.SchemeObject], types)

    def test_builtin_function_call_type(self) -> None:
        prog = sexp.parse("(number? 42)")
        analyzer = FunctionTypeAnalyzer({})
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
            {sexp.SSym('egg'): scheme_types.SchemeNum})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
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
        prog = sexp.parse("(if true 42 43)")
        analyzer = FunctionTypeAnalyzer({})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeBool,
            scheme_types.SchemeNum,
            scheme_types.SchemeNum,
            scheme_types.SchemeNum,
        ]
        self.assertEqual(expected, types)

    def test_conditional_different_type_branches(self) -> None:
        prog = sexp.parse("(if true 42 false)")
        analyzer = FunctionTypeAnalyzer({})
        analyzer.visit(prog)

        types = list(analyzer.get_expr_types().values())
        expected = [
            scheme_types.SchemeBool,
            scheme_types.SchemeNum,
            scheme_types.SchemeBool,
            scheme_types.SchemeObject,
        ]
        self.assertEqual(expected, types)
