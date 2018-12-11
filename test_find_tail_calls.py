import unittest
from typing import List

import emit_IR
import scheme_types
import sexp
from find_tail_calls import TailCallData, TailCallFinder


class TailCallFinderTestCase(unittest.TestCase):
    finder: TailCallFinder

    def setUp(self) -> None:
        self.finder = TailCallFinder()

    def test_basic_tail_call(self) -> None:
        prog = sexp.parse('(define (vacuous-tail) (vacuous-tail))')
        self.finder.visit(prog)
        self.assertEqual(1, len(self.finder.tail_calls))
        self.assert_symbol_in_tail_calls(
            sexp.SSym('vacuous-tail'), self.finder.tail_calls)

    def test_tail_call_in_conditional_then(self) -> None:
        prog = sexp.parse(
            '(define (vacuous-tail) (if true (vacuous-tail) false))')
        self.finder.visit(prog)
        self.assertEqual(1, len(self.finder.tail_calls))
        self.assert_symbol_in_tail_calls(
            sexp.SSym('vacuous-tail'), self.finder.tail_calls)

    def test_tail_call_in_begin(self) -> None:
        prog = sexp.parse(
            '(define (vacuous-tail) (begin 42 43 (vacuous-tail)))')
        self.finder.visit(prog)
        self.assertEqual(1, len(self.finder.tail_calls))
        self.assert_symbol_in_tail_calls(
            sexp.SSym('vacuous-tail'), self.finder.tail_calls)

    def test_tail_call_in_conditional_else(self) -> None:
        prog = sexp.parse(
            '(define (vacuous-tail) (if true false (vacuous-tail)))')
        self.finder.visit(prog)
        self.assertEqual(1, len(self.finder.tail_calls))
        self.assert_symbol_in_tail_calls(
            sexp.SSym('vacuous-tail'), self.finder.tail_calls)

    def test_tail_call_in_conditional_both_branches(self) -> None:
        prog = sexp.parse(
            '(define (vacuous-tail) (if true (vacuous-tail) (vacuous-tail)))')
        self.finder.visit(prog)
        self.assertEqual(2, len(self.finder.tail_calls))
        self.assert_symbol_in_tail_calls(
            sexp.SSym('vacuous-tail'), self.finder.tail_calls)

    def test_tail_call_in_one_branch_linear_call_in_other(self) -> None:
        prog = sexp.parse('''
            (define (vacuous-tail)
                (if true (+ 1 (vacuous-tail)) (vacuous-tail))
            )''')
        self.finder.visit(prog)
        self.assertEqual(1, len(self.finder.tail_calls))
        self.assert_symbol_in_tail_calls(
            sexp.SSym('vacuous-tail'), self.finder.tail_calls)

    def test_recursive_call_not_last_expr(self) -> None:
        prog = sexp.parse(
            '(define (not-tail) (not-tail) (if true true false) (not-tail))')
        self.finder.visit(prog)
        self.assertEqual(1, len(self.finder.tail_calls))

    def test_tail_call_has_recusive_call_as_args(self) -> None:
        prog = sexp.parse(
            '(define (tail-ish arg1 arg2) (tail-ish 42 (tail-ish 42 43)))')
        self.finder.visit(prog)
        self.assertEqual(1, len(self.finder.tail_calls))

    def test_fib_tail(self) -> None:
        prog = sexp.parse('''
            (define (fib-tail n)
                (fib-tail-impl n 0 1)
            )

            (define (fib-tail-impl n first second)
                (if (= n 0)
                    first
                    (if (= n 1)
                        second
                        (fib-tail-impl (- n 1) second (+ first second))
                    )
                )
            )''')

        self.finder.visit(prog)
        self.assertEqual(1, len(self.finder.tail_calls))
        self.assert_symbol_in_tail_calls(
            sexp.SSym('fib-tail-impl'), self.finder.tail_calls)

    def assert_symbol_in_tail_calls(self, sym: sexp.SSym,
                                    tail_calls: List[TailCallData]) -> None:
        for call_data in tail_calls:
            if sym == call_data.call.func:
                return

        self.fail(f'Symbol: {sym} not found in {tail_calls}')
