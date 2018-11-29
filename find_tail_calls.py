from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, cast

import sexp
from visitor import Visitor


@dataclass
class TailCallData:
    call: sexp.SCall
    func_params: List[sexp.SSym] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(id(self.call))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TailCallData):
            return False
        return hash(self) == hash(other)


class TailCallFinder(Visitor):
    def __init__(self) -> None:
        self._tail_calls: List[TailCallData] = []

        self._current_function: Optional[sexp.SFunction] = None

    @property
    def tail_calls(self) -> List[TailCallData]:
        return list(self._tail_calls)

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
            self._tail_calls.append(
                TailCallData(call, list(self._current_function.params)))


class RecursiveCallFinder(Visitor):
    def __init__(self, func: sexp.SFunction) -> None:
        self._func = func
        self.recursive_calls: List[sexp.SCall] = []

    def visit_SCall(self, call: sexp.SCall) -> None:
        if (isinstance(call.func, sexp.SSym)
                and call.func == self._func.name):
            self.recursive_calls.append(call)
