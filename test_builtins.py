import unittest
from typing import Dict

import errors
import runner
from runner import run
from scheme import SBool, SNum, SSym, Value


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
        pass

    def test_prelude(self) -> None:
        pass
