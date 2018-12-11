from typing import List, Union

from sexp import (Nil, Quote, SBegin, SBool, SCall, SConditional, SExp,
                  SFunction, SNum, SPair, SSym, SVect)


class Visitor:
    """Base class for traversing scheme programs."""

    def visit(self, expr: Union[List[SExp], SExp]) -> None:
        if isinstance(expr, list):
            for item in expr:
                self.visit(item)
        else:
            self.visit_SExp(expr)

    def visit_SExp(self, expr: SExp) -> None:
        getattr(self, f'visit_{type(expr).__name__}')(expr)

    def visit_SNum(self, num: SNum) -> None:
        pass

    def visit_SBool(self, sbool: SBool) -> None:
        pass

    def visit_SSym(self, sym: SSym) -> None:
        pass

    def visit_SVect(self, vect: SVect) -> None:
        for item in vect.items:
            self.visit(item)

    def visit_SPair(self, pair: SPair) -> None:
        self.visit(pair.first)
        if pair.second is not Nil:
            self.visit(pair.second)

    def visit_Quote(self, quote: Quote) -> None:
        self.visit(quote.expr)

    def visit_SBegin(self, begin: SBegin) -> None:
        for expr in begin.exprs:
            self.visit(expr)

    def visit_SFunction(self, func: SFunction) -> None:
        if not func.is_lambda:
            self.visit(func.name)

        for param in func.params:
            self.visit(param)

        for expr in func.body:
            self.visit(expr)

    def visit_SCall(self, call: SCall) -> None:
        self.visit(call.func)
        self.visit(call.args)

    def visit_SConditional(self, cond: SConditional) -> None:
        self.visit(cond.test)
        self.visit(cond.then_expr)
        self.visit(cond.else_expr)
