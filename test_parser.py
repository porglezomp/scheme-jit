import unittest

import scheme
from scheme import SSym, SVect, SNum, SPair, Nil


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
            scheme.parse("(if (< x 0) 1 2)"),
            [
                scheme.to_slist([
                    SSym("if"),
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


if __name__ == '__main__':
    unittest.main()
