import unittest


class FunctionTypeAnalyzerTestCase(unittest.TestCase):
    def test_quoted_symbol(self) -> None:
        self.fail()

    def test_quoted_list(self) -> None:
        self.fail()

    def test_num_literal(self) -> None:
        self.fail()

    def test_bool_literal(self) -> None:
        self.fail()

    def test_sym_literal_not_function(self) -> None:
        self.fail()

    def test_sym_literal_is_local_function(self) -> None:
        self.fail()

    def test_sym_literal_is_builtin_function(self) -> None:
        self.fail()

    def test_function_def(self) -> None:
        self.fail()

    def test_function_call_type_unknown(self) -> None:
        self.fail()

    def test_builtin_function_call_type(self) -> None:
        self.fail()

    def test_conditional_same_type_branches(self) -> None:
        self.fail()

    def test_conditional_different_type_branches(self) -> None:
        self.fail()

    def test_symbol_type_narrowed_by_conditional(self) -> None:
        self.fail()

    def test_symbol_type_narrowed_by_assert(self) -> None:
        self.fail()
