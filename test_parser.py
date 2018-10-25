import unittest

import scheme
from scheme import SSym, SVect, SNum, SPair, Nil, SConditional, SFunction


class ParserTestCase(unittest.TestCase):
    def test_to_slist(self):
        self.assertEqual(
            scheme.to_slist([SSym(1), SSym(2), SSym(3)]),
            SPair(
                SSym(1),
                SPair(
                    SSym(2),
                    SPair(
                        SSym(3),
                        Nil,
                    ),
                ),
            )
        )

    def test_parse_atoms(self):
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

    def test_parse_list(self):
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

    def test_vector(self):
        self.assertEqual(scheme.parse("[]"), [SVect([])])
        self.assertEqual(
            scheme.parse("[1 [2 [3 []]]]"),
            [SVect([SNum(1), SVect([SNum(2), SVect([SNum(3), SVect([])])])])]
        )

    def test_quote(self):
        self.assertEqual(
            scheme.parse("'(1 2 3)"), scheme.parse("(quote (1 2 3))"))

        # fixme?
        # self.assertEqual(str(scheme.parse("(quote (1 2 3))")[0]), "'(1 2 3)")

    def test_conditional(self):
        prog = '(if true 42 43) (if false 44 45)'
        self.assertEqual(
            [
                SConditional(SSym('true'), SNum(42), SNum(43)),
                SConditional(SSym('false'), SNum(44), SNum(45)),
            ],
            scheme.parse(prog)
        )

    def test_function_def(self):
        prog = '(define (funcy spam egg) (+ spam egg)) (funcy 42 43)'
        self.assertEqual(
            [
                SFunction(
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


if __name__ == '__main__':
    unittest.main()
