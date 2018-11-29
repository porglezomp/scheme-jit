from typing import List, Optional, Set, cast

import sexp
from visitor import Visitor


class TailCallFinder(Visitor):
    def __init__(self) -> None:
        self._tail_calls: Set[sexp.SCall] = set()

        self._current_function: Optional[sexp.SFunction] = None

    @property
    def tail_calls(self) -> Set[sexp.SCall]:
        return set(self._tail_calls)

    def visit_SFunction(self, func: sexp.SFunction) -> None:
        if func.is_lambda:
            # We're only looking for statically-detectable tail recursion.
            return

        body_exprs = list(func.body)
        assert len(body_exprs) > 0, 'Function bodies must not be empty'

        for body_expr in body_exprs[:-1]:
            recursive_call_finder = RecursiveCallFinder(func)
            recursive_call_finder.visit(body_expr)
            if recursive_call_finder.recursive_calls:
                return

        old_current_func = self._current_function
        self._current_function = func
        self.visit(body_exprs[-1])
        self._current_function = old_current_func

    def visit_SCall(self, call: sexp.SCall) -> None:
        if self._current_function is None:
            return

        if (isinstance(call.func, sexp.SSym)
                and call.func != self._current_function.name):
            return

        recursive_call_finder = RecursiveCallFinder(self._current_function)
        for arg_expr in call.args:
            recursive_call_finder.visit(arg_expr)
        if not recursive_call_finder.recursive_calls:
            self._tail_calls.add(call)


class RecursiveCallFinder(Visitor):
    def __init__(self, func: sexp.SFunction) -> None:
        self._func = func
        self.recursive_calls: List[sexp.SCall] = []

    def visit_SCall(self, call: sexp.SCall) -> None:
        if (isinstance(call.func, sexp.SSym)
                and call.func == self._func.name):
            self.recursive_calls.append(call)
