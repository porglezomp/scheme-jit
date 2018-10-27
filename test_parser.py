import unittest

import scheme
from scheme import SSym, SVect, SNum, SPair, Nil, SConditional, SFunction


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
            scheme.parse("(1 2 3)"),
            [scheme.to_slist([SNum(1), SNum(2), SNum(3)])],
        )
        self.assertEqual(
            scheme.parse("(spam (< x 0) 1 2)"),
            [
                scheme.to_slist([
                    SSym("spam"),
                    scheme.to_slist([SSym("<"), SSym("x"), SNum(0)]),
                    SNum(1),
                    SNum(2),
                ]),
            ]
        )

    def test_vector(self) -> None:
        self.assertEqual(scheme.parse("[]"), [SVect([])])
        self.assertEqual(
            scheme.parse("[1 [2 [3 []]]]"),
            [SVect([SNum(1), SVect([SNum(2), SVect([SNum(3), SVect([])])])])]
        )

    def test_quote(self) -> None:
        self.assertEqual(
            scheme.parse("'(1 2 3)"), scheme.parse("(quote (1 2 3))"))

        # fixme?
        # self.assertEqual(str(scheme.parse("(quote (1 2 3))")[0]), "'(1 2 3)")

    def test_conditional(self) -> None:
        prog = '(if true 42 43) (if false 44 45)'
        self.assertEqual(
            [
                SConditional(SSym('true'), SNum(42), SNum(43)),
                SConditional(SSym('false'), SNum(44), SNum(45)),
            ],
            scheme.parse(prog)
        )

    def test_function_def(self) -> None:
        prog = '(define (funcy spam egg) (+ spam egg)) (funcy 42 43)'
        self.assertEqual(
            [
                SFunction(
                    SSym('funcy'),
                    scheme.to_slist([SSym('spam'), SSym('egg')]),
                    scheme.to_slist(
                        [scheme.to_slist(
                            [SSym('+'), SSym('spam'), SSym('egg')])])
                ),
                scheme.to_slist(
                    [SSym('funcy'), SNum(42), SNum(43)]
                )
            ],
            scheme.parse(prog)
        )

    def test_lambda(self) -> None:
        prog = '(lambda (spam egg) (+ spam egg)) (lambda () 42)'
        self.assertEqual(
            [
                SFunction(
                    SSym('lambda0'),
                    scheme.to_slist([SSym('spam'), SSym('egg')]),
                    scheme.to_slist(
                        [scheme.to_slist(
                            [SSym('+'), SSym('spam'), SSym('egg')])]
                    ),
                    is_lambda=True
                ),
                SFunction(
                    SSym('lambda1'),
                    Nil,
                    scheme.to_slist([SNum(42)]),
                    is_lambda=True
                ),
            ],
            scheme.parse(prog)
        )


if __name__ == '__main__':
    unittest.main()
