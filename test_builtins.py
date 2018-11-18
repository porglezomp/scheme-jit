import unittest
from typing import Dict

import errors
import runner
from runner import run
from scheme import SBool, SNum, SSym, SVect, Value


class BuiltinsTestCase(unittest.TestCase):
    def test_intrinsics(self) -> None:
        env: Dict[SSym, Value] = {}
        runner.add_intrinsics(env)

        self.assertEqual(run(env, '(inst/typeof 42)'), SSym('number'))
        self.assertEqual(run(env, '(inst/typeof [1 2 3])'), SSym('vector'))
        self.assertEqual(run(env, "(inst/typeof 'hi)"), SSym('symbol'))
        with self.assertRaises(errors.Trap, msg="(trap)"):
            run(env, '(inst/trap)')
        self.assertEqual(run(env, '(inst/length (inst/alloc 42))'), SNum(42))
        self.assertEqual(
            run(env, '(inst/load (inst/store (inst/alloc 8) 3 42) 3)'),
            SNum(42)
        )

        self.assertEqual(run(env, '(inst/+ 18 24)'), SNum(18 + 24))
        self.assertEqual(run(env, '(inst/- 18 24)'), SNum(18 - 24))
        self.assertEqual(run(env, '(inst/* 18 24)'), SNum(18 * 24))
        self.assertEqual(run(env, '(inst// 18 24)'), SNum(18 // 24))
        self.assertEqual(run(env, '(inst/% 18 24)'), SNum(18 % 24))

        self.assertEqual(run(env, '(inst/number= 18 18)'), SBool(True))
        self.assertEqual(run(env, '(inst/number= 18 -18)'), SBool(False))
        self.assertEqual(run(env, "(inst/symbol= 'hi 'hey)"), SBool(False))
        self.assertEqual(run(env, "(inst/symbol= 'hi 'hi)"), SBool(True))
        self.assertEqual(run(env, "(inst/pointer= [] 0)"), SBool(False))
        self.assertEqual(
            run(env, '((lambda (x) (inst/pointer= x x)) [1])'),
            SBool(True)
        )
        self.assertEqual(run(env, '(inst/number< -1 0)'), SBool(True))

    def test_builtins(self) -> None:
        env: Dict[SSym, Value] = {}
        runner.add_intrinsics(env)
        runner.add_builtins(env)

        with self.assertRaises(errors.Trap, msg="(trap)"):
            run(env, '(trap)')
        with self.assertRaises(errors.Trap, msg="(trap)"):
            run(env, '(assert false)')
        self.assertEqual(run(env, '(typeof 42)'), SSym('number'))
        self.assertEqual(run(env, '(typeof [1 2 3])'), SSym('vector'))
        self.assertEqual(run(env, "(typeof 'hi)"), SSym('symbol'))
        self.assertEqual(run(env, '(not true)'), SBool(False))
        self.assertEqual(run(env, '(not false)'), SBool(True))
        with self.assertRaises(errors.Trap, msg="(trap)"):
            run(env, '(not 42)')

        self.assertEqual(run(env, '(number? 42)'), SBool(True))
        self.assertEqual(run(env, '(number? [])'), SBool(False))
        self.assertEqual(run(env, "(symbol? 'hi)"), SBool(True))
        self.assertEqual(run(env, '(symbol? 42)'), SBool(False))
        self.assertEqual(run(env, '(vector? [])'), SBool(True))
        self.assertEqual(run(env, '(vector? 42)'), SBool(False))
        self.assertEqual(run(env, '(function? (lambda () []))'), SBool(True))
        self.assertEqual(run(env, '(function? 42)'), SBool(False))
        self.assertEqual(run(env, '(bool? true)'), SBool(True))
        self.assertEqual(run(env, '(bool? false)'), SBool(True))
        self.assertEqual(run(env, '(bool? 42)'), SBool(False))
        self.assertEqual(run(env, '(pair? 42)'), SBool(False))
        self.assertEqual(run(env, '(pair? [])'), SBool(False))
        self.assertEqual(run(env, '(pair? [1])'), SBool(False))
        self.assertEqual(run(env, '(pair? [1 2])'), SBool(True))
        self.assertEqual(run(env, '(nil? [1 2])'), SBool(False))
        self.assertEqual(run(env, '(nil? [])'), SBool(True))

        self.assertEqual(run(env, '(+ 39 13)'), SNum(39 + 13))
        self.assertEqual(run(env, '(- 39 13)'), SNum(39 - 13))
        self.assertEqual(run(env, '(* 39 13)'), SNum(39 * 13))
        self.assertEqual(run(env, '(/ 39 13)'), SNum(39 // 13))
        self.assertEqual(run(env, '(% 39 13)'), SNum(39 % 13))

        for op in '+-*/%':
            with self.assertRaises(errors.Trap, msg="(trap)"):
                run(env, f'({op} 39 [])')
            with self.assertRaises(errors.Trap, msg="(trap)"):
                run(env, f'({op} [] 13)')

        with self.assertRaises(errors.Trap, msg="(trap)"):
            run(env, '(/ 1 0)')
        with self.assertRaises(errors.Trap, msg="(trap)"):
            run(env, '(% 1 0)')

        with self.assertRaises(errors.Trap, msg="(trap)"):
            run(env, '(symbol= 1 0)')
        self.assertEqual(run(env, "(symbol= 'a 'b)"), SBool(False))
        self.assertEqual(run(env, "(symbol= 'a 'a)"), SBool(True))
        # I don't want to test the other foo= functions, they're very similar

        self.assertEqual(run(env, '(vector-length [1 2 3])'), SNum(3))
        self.assertEqual(run(env, '(vector-index [1 2 3] 1)'), SNum(2))
        self.assertEqual(
            run(env, '(vector-set! [1 2 3] 1 42)'),
            SVect([SNum(1), SNum(42), SNum(3)])
        )

        self.assertEqual(
            run(env, '(vector-make 4 9)'),
            SVect([SNum(9), SNum(9), SNum(9), SNum(9)])
        )

    def test_prelude(self) -> None:
        env: Dict[SSym, Value] = {}
        runner.add_intrinsics(env)
        runner.add_builtins(env)
        runner.add_prelude(env)

        self.assertEqual(
            run(env, "(= (cons 42 (cons 13 (cons 'a []))) '(42 13 a))"),
            SBool(True)
        )
        self.assertEqual(
            run(env, "(= [1 2 [3 4 5] 6 [[7]]] [1 2 [3 4 5] 6 [[7]]])"),
            SBool(True)
        )
        self.assertEqual(
            run(env, "(= [1 2 [3 4 5] 6 [[7]]] [1 2 [3 4 5] 6 [[7]]])"),
            SBool(True)
        )
