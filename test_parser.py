import unittest

import sexp
from sexp import (Nil, Quote, SBool, SCall, SConditional, SFunction, SNum,
                  SPair, SSym, SVect)


class ParserTestCase(unittest.TestCase):
    def test_to_slist(self) -> None:
        self.assertEqual(
            sexp.to_slist([SNum(1), SNum(2), SNum(3)]),
            SPair(
                SNum(1),
                SPair(
                    SNum(2),
                    SPair(
                        SNum(3),
                        Nil,
                    ),
                ),
            )
        )

    def test_parse_atoms(self) -> None:
        self.assertEqual(sexp.parse("hi"), [SSym("hi")])
        self.assertEqual(
            sexp.parse("hi hey hoi"),
            [
                SSym("hi"),
                SSym("hey"),
                SSym("hoi"),
            ]
        )
        self.assertEqual(sexp.parse("42 foo"), [SNum(42), SSym("foo")])

    def test_parse_list(self) -> None:
        self.assertEqual(sexp.parse("()"), [Nil])
        self.assertEqual(
            sexp.parse("(func 2 3)"),
            [SCall(SSym('func'), [SNum(2), SNum(3)])],
        )

    def test_vector(self) -> None:
        self.assertEqual(sexp.parse("[]"), [SVect([])])
        self.assertEqual(
            sexp.parse("[1 [2 [3 []]]]"),
            [SVect([SNum(1), SVect([SNum(2), SVect([SNum(3), SVect([])])])])]
        )

    def test_quote(self) -> None:
        self.assertEqual([Quote(SSym('spam'))], sexp.parse("'spam"))

        self.assertEqual([Quote(Nil)], sexp.parse("'()"))
        self.assertEqual([Quote(Nil)], sexp.parse("(quote ())"))

        self.assertEqual([
                Quote(
                    sexp.to_slist(
                        [SSym('if'), SBool(True), SNum(2), SNum(3)]
                    )
                )
            ],
            sexp.parse("'(if true 2 3)"))

        self.assertEqual([Quote(sexp.to_slist([SNum(1), SNum(2), SNum(3)]))],
                         sexp.parse("(quote (1 2 3))"))

        self.assertEqual(
            sexp.parse("'(1 2 3)"), sexp.parse("(quote (1 2 3))"))

        self.assertEqual(str(sexp.parse("(quote (1 2 3))")[0]), "'(1 2 3)")

    def test_conditional(self) -> None:
        prog = '(if true 42 43) (if false 44 45)'
        self.assertEqual(
            [
                SConditional(SBool(True), SNum(42), SNum(43)),
                SConditional(SBool(False), SNum(44), SNum(45)),
            ],
            sexp.parse(prog)
        )

    def test_function_def(self) -> None:
        prog = '(define (funcy spam egg) (+ spam egg)) (funcy 42 43)'
        self.assertEqual(
            [
                SFunction(
                    SSym('funcy'),
                    [SSym('spam'), SSym('egg')],
                    sexp.to_slist([
                        SCall(SSym('+'), [SSym('spam'), SSym('egg')])
                    ])
                ),
                SCall(SSym('funcy'), [SNum(42), SNum(43)])
            ],
            sexp.parse(prog)
        )

    def test_lambda(self) -> None:
        prog = '(lambda (spam egg) (+ spam egg)) (lambda () 42)'
        self.assertEqual(
            [
                SFunction(
                    SSym('__lambda0'),
                    [SSym('spam'), SSym('egg')],
                    sexp.to_slist([
                        SCall(SSym('+'), [SSym('spam'), SSym('egg')])
                    ]),
                    is_lambda=True
                ),
                SFunction(
                    SSym('__lambda1'),
                    [],
                    sexp.to_slist([SNum(42)]),
                    is_lambda=True
                ),
            ],
            sexp.parse(prog)
        )

    def test_lambda_called_inline(self) -> None:
        self.maxDiff = None
        prog = '((lambda (spam egg) (+ spam egg)) 42 43)'
        self.assertEqual(
            [
                SCall(
                    SFunction(
                        SSym('__lambda0'),
                        [SSym('spam'), SSym('egg')],
                        sexp.to_slist([
                            SCall(SSym('+'), [SSym('spam'), SSym('egg')])
                        ]),
                        is_lambda=True
                    ),
                    [SNum(42), SNum(43)]
                )
            ],
            sexp.parse(prog)
        )

    def test_comments(self) -> None:
        prog = """
        ;;; We want to define a cool function here!
        (define ; hi
          ;; A function name
          (cool-func x y) ; wow
          ;; branches are cheaper than subtraction, right? :P
          (if (= x y)
            0
            (- x y)))
        """
        self.assertEqual(
            [
                SFunction(
                    SSym('cool-func'),
                    [SSym('x'), SSym('y')],
                    sexp.to_slist([
                        SConditional(
                            SCall(SSym('='), [SSym('x'), SSym('y')]),
                            SNum(0),
                            SCall(SSym('-'), [SSym('x'), SSym('y')])
                        )
                    ])
                )
            ],
            sexp.parse(prog)
        )


if __name__ == '__main__':
    unittest.main()
