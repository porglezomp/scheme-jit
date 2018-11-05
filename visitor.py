from typing import List, Union

from scheme import Nil, SConditional, SExp, SFunction, SNum, SPair, SSym, SVect


class Visitor:
    """
    Base class for traversing scheme programs.

    >>> import scheme
    >>> prog = '(define (spam egg) (if (> spam egg) [1 true] [2 false]))'
    >>> parsed = scheme.parse(prog)
    >>> _DoctestVisitor().visit(parsed)
    SExp
    SFunction
    SExp
    spam
    SExp
    egg
    SExp
    SConditional
    SExp
    SPair
    SExp
    >
    SExp
    SPair
    SExp
    spam
    SExp
    SPair
    SExp
    egg
    SExp
    SVect
    SExp
    1
    SExp
    true
    SExp
    SVect
    SExp
    2
    SExp
    false

    >>> prog = '(lambda (spam) 42)'
    >>> parsed = scheme.parse(prog)
    >>> _DoctestVisitor().visit(parsed)
    SExp
    SFunction
    SExp
    spam
    SExp
    42

    """

    def visit(self, expr: Union[List[SExp], SExp]) -> None:
        if isinstance(expr, list):
            for item in expr:
                self.visit(item)

            return

        self.visit_SExp(expr)

        if isinstance(expr, SNum):
            self.visit_SNum(expr)
        elif isinstance(expr, SSym):
            self.visit_SSym(expr)
        elif isinstance(expr, SVect):
            self.visit_SVect(expr)
        elif isinstance(expr, SPair):
            self.visit_SPair(expr)
        elif isinstance(expr, SFunction):
            self.visit_SFunction(expr)
        elif isinstance(expr, SConditional):
            self.visit_SConditional(expr)

    def visit_SExp(self, expr: SExp) -> None:
        pass

    def visit_SNum(self, num: SNum) -> None:
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

    def visit_SFunction(self, func: SFunction) -> None:
        if not func.is_lambda:
            self.visit(func.name)

        for param in func.params:
            self.visit(param)

        for expr in func.body:
            self.visit(expr)

    def visit_SConditional(self, cond: SConditional) -> None:
        self.visit(cond.test)
        self.visit(cond.then_expr)
        self.visit(cond.else_expr)


class _DoctestVisitor(Visitor):
    def visit_SExp(self, expr: SExp) -> None:
        print('SExp')
        super().visit_SExp(expr)

    def visit_SNum(self, num: SNum) -> None:
        print(str(num))
        super().visit_SNum(num)

    def visit_SSym(self, sym: SSym) -> None:
        print(str(sym))
        super().visit_SSym(sym)

    def visit_SVect(self, vect: SVect) -> None:
        print('SVect')
        super().visit_SVect(vect)

    def visit_SPair(self, pair: SPair) -> None:
        print('SPair')
        super().visit_SPair(pair)

    def visit_SFunction(self, func: SFunction) -> None:
        print('SFunction')
        super().visit_SFunction(func)

    def visit_SConditional(self, cond: SConditional) -> None:
        print('SConditional')
        super().visit_SConditional(cond)
