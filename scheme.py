from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Iterator, List, Optional, Sequence, Tuple,
                    Union, cast)

import bytecode


class SExp:
    """An s-expression base class"""
    ...


class Value(SExp):
    """An s-expression that's a valid run-time object."""
    @abstractmethod
    def type_name(self) -> SSym:
        ...

    @abstractmethod
    def address(self) -> int:
        ...


@dataclass(frozen=True, order=True)
class SNum(Value):
    """A lisp number"""
    value: int

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(self.value)

    def type_name(self) -> SSym:
        return SSym('number')

    def address(self) -> int:
        return self.value


@dataclass(frozen=True)
class SBool(Value):
    """A lisp boolean"""
    value: bool

    def __str__(self) -> str:
        return str(self.value)

    def type_name(self) -> SSym:
        return SSym('bool')


@dataclass(frozen=True)
class SSym(Value):
    """A lisp symbol"""
    name: str

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def type_name(self) -> SSym:
        return SSym('symbol')

    def address(self) -> int:
        raise Exception("Should not take the address of a symbol")


@dataclass(frozen=True)
class SVect(Value):
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

    def type_name(self) -> SSym:
        return SSym('vector')

    def address(self) -> int:
        return id(self.items)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Quote(SExp):
    """A quoted list"""
    slist: SList

    def __str__(self) -> str:
        return f"'{str(self.slist)}"


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
class SFunction(Value):
    name: SSym
    params: List[SSym]
    body: SList
    code: Optional[bytecode.Function] = None
    is_lambda: bool = False

    def type_name(self) -> SSym:
        return SSym('function')

    def address(self) -> int:
        return id(self.code)


@dataclass(frozen=True)
class SCall(SExp):
    func: Union[SSym, SFunction]
    args: List[SExp]


@dataclass(frozen=True)
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
            return parse_quote(tokens[1:])
        elif tokens[0] == '[':
            vector = []
            tokens = tokens[1:]
            while tokens[0] != ']':
                item, tokens = parse(tokens)
                vector.append(item)
            return SVect(vector), tokens[1:]
        elif tokens[0] == '(':
            tokens = tokens[1:]

            if tokens[0] == ')':
                return Nil, tokens[1:]

            parsed_first, tokens = parse(tokens)
            if parsed_first == SSym('if'):
                return parse_conditional(tokens)

            if parsed_first == SSym('define'):
                return parse_define(tokens)

            if parsed_first == SSym('lambda'):
                return parse_lambda(tokens)

            if parsed_first == SSym('quote'):
                quote, tokens = parse_quote(tokens)
                return quote, tokens[1:]

            if isinstance(parsed_first, SFunction):
                assert parsed_first.is_lambda
                return parse_call(parsed_first, tokens)

            if isinstance(parsed_first, SSym):
                return parse_call(parsed_first, tokens)

            items, tokens = read_list_tail(tokens)

            return to_slist([parsed_first] + items), tokens[1:]

        elif tokens[0].isdigit():
            return SNum(int(tokens[0])), tokens[1:]
        elif tokens[0] in ('true', 'false'):
            return SBool(tokens[0] == 'true'), tokens[1:]
        else:
            return SSym(tokens[0]), tokens[1:]

    def parse_conditional(tokens: List[str]) -> Tuple[SExp, List[str]]:
        items, tokens = read_list_tail(tokens)
        assert len(items) == 3, 'Missing parts of conditional'
        return SConditional(
            items[0],
            items[1],
            items[2]
        ), tokens[1:]

    def parse_define(tokens: List[str]) -> Tuple[SExp, List[str]]:
        params, tokens = parse_function_params(tokens)
        assert len(params) >= 1, 'Missing function name'

        body, tokens = read_list_tail(tokens)
        return SFunction(
            params[0],
            params[1:],
            to_slist(body)
        ), tokens[1:]

    def parse_lambda(tokens: List[str]) -> Tuple[SExp, List[str]]:
        params, tokens = parse_function_params(tokens)

        body, tokens = read_list_tail(tokens)
        return SFunction(
            SSym(next(lambda_names)),
            params,
            to_slist(body),
            is_lambda=True
        ), tokens[1:]

    def parse_function_params(
            tokens: List[str]) -> Tuple[List[SSym], List[str]]:
        formals: List[SSym] = []

        assert tokens[0] == '(', 'Expected parameter list'
        tokens = tokens[1:]

        expr_list, tokens = read_list_tail(tokens)
        for item in expr_list:
            assert isinstance(item, SSym), 'Expected a symbol'
            formals.append(item)

        return formals, tokens[1:]

    def parse_call(func: Union[SSym, SFunction],
                   tokens: List[str]) -> Tuple[SExp, List[str]]:
        args, tokens = read_list_tail(tokens)
        return SCall(func, args), tokens[1:]

    def parse_quote(tokens: List[str]) -> Tuple[SExp, List[str]]:
        assert tokens[0] == '(', 'Expected list'
        tokens = tokens[1:]

        list_tail, tokens = read_list_tail(tokens)
        return Quote(to_slist(list_tail)), tokens[1:]

    def read_list_tail(tokens: List[str]) -> Tuple[List[SExp], List[str]]:
        items = []

        while tokens[0] != ')':
            item, tokens = parse(tokens)
            items.append(item)

        return items, tokens

    results = []
    while tokens:
        result, tokens = parse(tokens)
        results.append(result)
    return results


def lambda_name_generator() -> Iterator[str]:
    n = 0
    while True:
        yield f'__lambda{n}'
        n += 1
