from __future__ import annotations

from dataclasses import dataclass
from typing import (TYPE_CHECKING, Iterator, List, Optional, Sequence, Tuple,
                    Union, cast)

import bytecode

if TYPE_CHECKING:
    from environment import Environment


class SExp:
    """An s-expression base class"""
    def __init__(self) -> None:
        self._environment: Optional[Environment] = None

    @property
    def environment(self) -> Environment:
        if self._environment is None:
            raise Exception('Environment not set')

        return self._environment

    @environment.setter
    def environment(self, env: Environment) -> None:
        self._environment = env


@dataclass(order=True)
class SNum(SExp):
    """A lisp number"""
    value: int

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass
class SSym(SExp):
    """A lisp symbol"""
    name: str

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class SVect(SExp):
    """An n element vector

    >>> vect = SVect([
    ...     SNum(42), SSym('spam'),
    ...     SVect([SNum(43), SNum(44)])
    ... ])
    >>> str(vect)
    '[42 spam [43 44]]'
    >>>
    >>> str(SVect([]))
    '[]'
    """
    items: List[SExp]

    def __str__(self) -> str:
        return f"[{' '.join(str(i) for i in self.items)}]"


@dataclass
class SPair(SExp):
    """A scheme pair.

    >>> single_pair = SPair(SNum(42), SSym('spam'))
    >>> str(single_pair)
    '(42 . spam)'
    >>> list(single_pair)
    [SNum(value=42), SSym(name='spam')]

    >>> nested_pair = SPair(SNum(42), SPair(SSym('egg'), SSym('spam')))
    >>> str(nested_pair)
    '(42 . (egg . spam))'
    >>> list(nested_pair)
    [SNum(value=42), SSym(name='egg'), SSym(name='spam')]

    >>> s_list = SPair(SNum(42), SPair(SSym('egg'), Nil))
    >>> str(s_list)
    '(42 egg)'
    >>> list(s_list)
    [SNum(value=42), SSym(name='egg')]
    """
    first: SExp
    second: SExp

    def is_list(self) -> bool:
        if self.second is Nil:
            return True

        if not isinstance(self.second, SPair):
            return False

        return self.second.is_list()

    def __iter__(self) -> PairIterator:
        return PairIterator(self)

    def __str__(self) -> str:
        if self.is_list():
            return f"({' '.join((str(item) for item in self))})"

        return f"({str(self.first)} . {str(self.second)})"


class PairIterator:
    def __init__(self, pair: SPair):
        self._expr: SExp = pair

    def __next__(self) -> SExp:
        if self._expr is Nil:
            raise StopIteration

        if not isinstance(self._expr, SPair):
            val = self._expr
            self._expr = Nil
            return val

        val = self._expr.first
        self._expr = self._expr.second

        return val

    def __iter__(self) -> PairIterator:
        return self


class NilType(SExp):
    def __iter__(self) -> NilIterator:
        return self.NilIterator()

    class NilIterator:
        def __next__(self) -> SExp:
            raise StopIteration

        def __iter__(self) -> NilType.NilIterator:
            return self

    def __str__(self) -> str:
        return 'Nil'


Nil = NilType()

SList = Union[SPair, NilType]


def to_slist(x: Sequence[SExp]) -> SList:
    acc: SList = Nil
    for item in reversed(x):
        acc = SPair(item, acc)

    return acc


def make_bool(x: bool) -> SSym:
    """
    Returns a scheme boolean.

    >>> make_bool(True)
    SSym(name='true')
    >>> make_bool(False)
    SSym(name='false')
    """
    if x:
        return SSym('true')
    return SSym('false')


@dataclass
class SFunction(SExp):
    name: SSym
    formals: SList
    body: SList
    code: Optional[bytecode.Function] = None
    is_lambda: bool = False


@dataclass
class SConditional(SExp):
    test: SExp
    then_expr: SExp
    else_expr: SExp


def parse(x: str) -> List[SExp]:
    tokens = (
        x
        .replace('(', ' ( ')
        .replace(')', ' ) ')
        .replace('[', ' [ ')
        .replace(']', ' ] ')
        .replace("'", " ' ")
        .split()
    )

    lambda_names = lambda_name_generator()

    def parse(tokens: List[str]) -> Tuple[SExp, List[str]]:
        if not tokens:
            raise Exception("Parse Error")
        elif tokens[0] == "'":
            item, tokens = parse(tokens[1:])
            return to_slist([SSym("quote"), item]), tokens
        elif tokens[0] == '[':
            vector = []
            tokens = tokens[1:]
            while tokens[0] != ']':
                item, tokens = parse(tokens)
                vector.append(item)
            return SVect(vector), tokens[1:]
        elif tokens[0] == '(':
            items = []
            tokens = tokens[1:]
            while tokens[0] != ')':
                item, tokens = parse(tokens)
                items.append(item)

            if len(items) == 0:
                return to_slist(items), tokens[1:]

            if items[0] == SSym('if'):
                assert len(items) == 4, 'Missing parts of conditional'
                return SConditional(
                    items[1],
                    items[2],
                    items[3]
                ), tokens[1:]

            if items[0] == SSym('define'):
                assert len(items) >= 3, 'Missing parts of function def'
                assert isinstance(items[1], SPair), 'Expected formals list'

                assert items[1] is not Nil, 'Missing function name'
                func_name = items[1].first
                assert isinstance(func_name, SSym)

                formals = items[1].second
                assert isinstance(formals, SPair) or formals is Nil
                return SFunction(
                    func_name,
                    cast(SList, formals),
                    to_slist(items[2:])
                ), tokens[1:]

            if items[0] == SSym('lambda'):
                assert len(items) >= 3, 'Missing parts of lambda def'
                formals = items[1]
                assert (
                    isinstance(formals, SPair) or formals is Nil
                ), 'Expected formals list'

                return SFunction(
                    SSym(next(lambda_names)),
                    cast(SList, formals),
                    to_slist(items[2:]),
                    is_lambda=True
                ), tokens[1:]

            return to_slist(items), tokens[1:]
        elif tokens[0].isdigit():
            return SNum(int(tokens[0])), tokens[1:]
        else:
            return SSym(tokens[0]), tokens[1:]

    results = []
    while tokens:
        result, tokens = parse(tokens)
        results.append(result)
    return results


def lambda_name_generator() -> Iterator[str]:
    n = 0
    while True:
        yield f'lambda{n}'
        n += 1
