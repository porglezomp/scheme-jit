import unittest

import scheme
from scheme import (Nil, Quote, SBool, SCall, SConditional, SFunction, SNum,
                    SPair, SSym, SVect)


class ParserTestCase(unittest.TestCase):
    def test_to_slist(self) -> None:
        self.assertEqual(
            scheme.to_slist([SNum(1), SNum(2), SNum(3)]),
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
        self.assertEqual(scheme.parse("hi"), [SSym("hi")])
        self.assertEqual(
            scheme.parse("hi hey hoi"),
            [
                SSym("hi"),
                SSym("hey"),
                SSym("hoi"),
            ]
        )
        self.assertEqual(scheme.parse("42 foo"), [SNum(42), SSym("foo")])

    def test_parse_list(self) -> None:
        self.assertEqual(scheme.parse("()"), [Nil])
        self.assertEqual(
            scheme.parse("(func 2 3)"),
            [SCall(SSym('func'), [SNum(2), SNum(3)])],
        )

    def test_vector(self) -> None:
        self.assertEqual(scheme.parse("[]"), [SVect([])])
        self.assertEqual(
            scheme.parse("[1 [2 [3 []]]]"),
            [SVect([SNum(1), SVect([SNum(2), SVect([SNum(3), SVect([])])])])]
        )

    def test_quote(self) -> None:
        self.assertEqual([Quote(SSym('spam'))], scheme.parse("'spam"))

        self.assertEqual([Quote(Nil)], scheme.parse("'()"))
        self.assertEqual([Quote(Nil)], scheme.parse("(quote ())"))

        self.assertEqual([
                Quote(
                    scheme.to_slist(
                        [SSym('if'), SBool(True), SNum(2), SNum(3)]
                    )
                )
            ],
            scheme.parse("'(if true 2 3)"))

        self.assertEqual([Quote(scheme.to_slist([SNum(1), SNum(2), SNum(3)]))],
                         scheme.parse("(quote (1 2 3))"))

        self.assertEqual(
            scheme.parse("'(1 2 3)"), scheme.parse("(quote (1 2 3))"))

        self.assertEqual(str(scheme.parse("(quote (1 2 3))")[0]), "'(1 2 3)")

    def test_conditional(self) -> None:
        prog = '(if true 42 43) (if false 44 45)'
        self.assertEqual(
            [
                SConditional(SBool(True), SNum(42), SNum(43)),
                SConditional(SBool(False), SNum(44), SNum(45)),
            ],
            scheme.parse(prog)
        )

    def test_function_def(self) -> None:
        prog = '(define (funcy spam egg) (+ spam egg)) (funcy 42 43)'
        self.assertEqual(
            [
                SFunction(
                    SSym('funcy'),
                    [SSym('spam'), SSym('egg')],
                    scheme.to_slist([
                        SCall(SSym('+'), [SSym('spam'), SSym('egg')])
                    ])
                ),
                SCall(SSym('funcy'), [SNum(42), SNum(43)])
            ],
            scheme.parse(prog)
        )

    def test_lambda(self) -> None:
        prog = '(lambda (spam egg) (+ spam egg)) (lambda () 42)'
        self.assertEqual(
            [
                SFunction(
                    SSym('__lambda0'),
                    [SSym('spam'), SSym('egg')],
                    scheme.to_slist([
                        SCall(SSym('+'), [SSym('spam'), SSym('egg')])
                    ]),
                    is_lambda=True
                ),
                SFunction(
                    SSym('__lambda1'),
                    [],
                    scheme.to_slist([SNum(42)]),
                    is_lambda=True
                ),
            ],
            scheme.parse(prog)
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
                        scheme.to_slist([
                            SCall(SSym('+'), [SSym('spam'), SSym('egg')])
                        ]),
                        is_lambda=True
                    ),
                    [SNum(42), SNum(43)]
                )
            ],
            scheme.parse(prog)
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
                    scheme.to_slist([
                        SConditional(
                            SCall(SSym('='), [SSym('x'), SSym('y')]),
                            SNum(0),
                            SCall(SSym('-'), [SSym('x'), SSym('y')])
                        )
                    ])
                )
            ],
            scheme.parse(prog)
        )


if __name__ == '__main__':
    unittest.main()
