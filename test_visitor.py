import unittest
from typing import List, Tuple

import scheme
from scheme import (Nil, Quote, SBool, SCall, SConditional, SExp, SFunction,
                    SNum, SPair, SSym, SVect)
from visitor import Visitor


class VisitorTestCase(unittest.TestCase):
    def test_visit_vect(self) -> None:
        prog = scheme.parse('[1 spam true]')
        recorder = TraversalRecorder()
        recorder.visit(prog)

        expected = [
            'SVect',
            'SNum',
            'SSym',
            'SBool',
        ]

        self.assertEqual(expected, recorder.exprs)

        counter = ExpressionCounter()
        counter.visit(prog)
        self.assertEqual(len(expected), counter.num_exprs)

    def test_visit_quote(self) -> None:
        prog = scheme.parse('(quote (1 spam true))')
        recorder = TraversalRecorder()
        recorder.visit(prog)

        expected = [
            'Quote',
            'SPair',
            'SNum',
            'SPair',
            'SSym',
            'SPair',
            'SBool',
        ]

        self.assertEqual(expected, recorder.exprs)

        counter = ExpressionCounter()
        counter.visit(prog)
        self.assertEqual(len(expected), counter.num_exprs)

    def test_visit_function(self) -> None:
        prog = scheme.parse('(define (spam egg sausage) true egg)')
        recorder = TraversalRecorder()
        recorder.visit(prog)

        expected = [
            'SFunction',
            'SSym',
            'SSym',
            'SSym',
            'SBool',
            'SSym',
        ]

        self.assertEqual(expected, recorder.exprs)

        counter = ExpressionCounter()
        counter.visit(prog)
        self.assertEqual(len(expected), counter.num_exprs)

    def test_visit_lambda(self) -> None:
        prog = scheme.parse('(lambda (egg sausage) false egg)')
        recorder = TraversalRecorder()
        recorder.visit(prog)

        expected = [
            'SFunction',
            'SSym',
            'SSym',
            'SBool',
            'SSym',
        ]

        self.assertEqual(expected, recorder.exprs)

        counter = ExpressionCounter()
        counter.visit(prog)
        self.assertEqual(len(expected), counter.num_exprs)

    def test_visit_conditional(self) -> None:
        prog = scheme.parse('(if true 42 nope)')
        recorder = TraversalRecorder()
        recorder.visit(prog)

        expected = [
            'SConditional',
            'SBool',
            'SNum',
            'SSym',
        ]

        self.assertEqual(expected, recorder.exprs)

        counter = ExpressionCounter()
        counter.visit(prog)
        self.assertEqual(len(expected), counter.num_exprs)

    def test_visit_call(self) -> None:
        prog = scheme.parse('(define (spam egg sausage) egg) (spam 42 true)')
        recorder = TraversalRecorder()
        recorder.visit(prog)

        expected = [
            'SFunction',
            'SSym',
            'SSym',
            'SSym',
            'SSym',

            'SCall',
            'SSym',
            'SNum',
            'SBool',
        ]

        self.assertEqual(expected, recorder.exprs)

        counter = ExpressionCounter()
        counter.visit(prog)
        self.assertEqual(len(expected), counter.num_exprs)

    def test_visit_inline_called_lambda(self) -> None:
        prog = scheme.parse('((lambda (egg) egg) 42)')
        recorder = TraversalRecorder()
        recorder.visit(prog)

        expected = [
            'SCall',
            'SFunction',
            'SSym',
            'SSym',
            'SNum',
        ]

        self.assertEqual(expected, recorder.exprs)

        counter = ExpressionCounter()
        counter.visit(prog)
        self.assertEqual(len(expected), counter.num_exprs)


class TraversalRecorder(Visitor):
    def __init__(self) -> None:
        self.exprs: List[str] = []

    def visit_SNum(self, num: SNum) -> None:
        self.exprs.append('SNum')
        super().visit_SNum(num)

    def visit_SBool(self, sbool: SBool) -> None:
        self.exprs.append('SBool')
        super().visit_SBool(sbool)

    def visit_SSym(self, sym: SSym) -> None:
        self.exprs.append('SSym')
        super().visit_SSym(sym)

    def visit_SVect(self, vect: SVect) -> None:
        self.exprs.append('SVect')
        super().visit_SVect(vect)

    def visit_SPair(self, pair: SPair) -> None:
        self.exprs.append('SPair')
        super().visit_SPair(pair)

    def visit_Quote(self, quote: Quote) -> None:
        self.exprs.append('Quote')
        super().visit_Quote(quote)

    def visit_SFunction(self, func: SFunction) -> None:
        self.exprs.append('SFunction')
        super().visit_SFunction(func)

    def visit_SCall(self, call: SCall) -> None:
        self.exprs.append('SCall')
        super().visit_SCall(call)

    def visit_SConditional(self, cond: SConditional) -> None:
        self.exprs.append('SConditional')
        super().visit_SConditional(cond)


class ExpressionCounter(Visitor):
    def __init__(self) -> None:
        self.num_exprs = 0

    def visit_SExp(self, expr: SExp) -> None:
        self.num_exprs += 1
        super().visit_SExp(expr)
