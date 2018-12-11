from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence,
                    Tuple, Union, cast)

import bytecode
import scheme_types

if TYPE_CHECKING:
    import bytecode
    from scheme_types import TypeTuple


class SExp:
    """An s-expression base class"""
    ...


class Value(SExp):
    """An s-expression that's a valid run-time object."""
    def type_name(self) -> SSym:
        result = self.scheme_type().symbol()
        assert result
        return result

    @abstractmethod
    def address(self) -> int:
        ...

    @abstractmethod
    def scheme_type(self) -> scheme_types.SchemeObjectType:
        ...

    @abstractmethod
    def to_param(self) -> Optional[bytecode.Parameter]:
        ...


@dataclass(frozen=True, order=True)
class SNum(Value):
    """A lisp number"""
    value: int

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(self.value)

    def address(self) -> int:
        return self.value

    def scheme_type(self) -> scheme_types.SchemeNumType:
        return scheme_types.SchemeNum

    def to_param(self) -> bytecode.NumLit:
        return bytecode.NumLit(self)


@dataclass(frozen=True)
class SBool(Value):
    """A lisp boolean"""
    value: bool

    def __str__(self) -> str:
        return str(self.value)

    def address(self) -> int:
        return id(self.value)

    def scheme_type(self) -> scheme_types.SchemeBoolType:
        return scheme_types.SchemeBool

    def to_param(self) -> bytecode.BoolLit:
        return bytecode.BoolLit(self)


@dataclass(frozen=True)
class SSym(Value):
    """A lisp symbol"""
    name: str

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def address(self) -> int:
        return id(self.name)

    def scheme_type(self) -> scheme_types.SchemeSymType:
        return scheme_types.SchemeSym

    def to_param(self) -> bytecode.SymLit:
        return bytecode.SymLit(self)


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

    def address(self) -> int:
        return id(self.items)

    def scheme_type(self) -> scheme_types.SchemeVectType:
        return scheme_types.SchemeVectType(len(self.items))

    def to_param(self) -> Optional[bytecode.Parameter]:
        return None


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
    """A quoted expression"""
    expr: SExp

    def __str__(self) -> str:
        return f"'{str(self.expr)}"


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

    specializations: Dict[TypeTuple, bytecode.Function] = \
        field(default_factory=dict)

    def address(self) -> int:
        return id(self.code)

    def scheme_type(self) -> scheme_types.SchemeFunctionType:
        return scheme_types.SchemeFunctionType(len(self.params))

    def __str__(self) -> str:
        params = ''.join(' ' + p.name for p in self.params)
        return f"<function ({self.name}{params}) at {id(self):x}>"

    def get_specialized(self, types: Optional[TypeTuple]) -> bytecode.Function:
        assert self.code
        if types is None:
            return self.code
        return self.specializations.get(types, self.code)

    def to_param(self) -> bytecode.FuncLit:
        return bytecode.FuncLit(self)


@dataclass(frozen=True)
class SCall(SExp):
    func: SExp
    args: List[SExp]


@dataclass(frozen=True)
class SConditional(SExp):
    test: SExp
    then_expr: SExp
    else_expr: SExp


def parse(x: str) -> List[SExp]:
    # Remove line comments. Since we don't have string literals,
    # we don't need fancier than this!
    x = '\n'.join(l.split(';')[0] for l in x.split('\n'))
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

    def is_number(x: str) -> bool:
        return x.isdecimal() or (x.startswith('-') and x[1:].isdecimal())

    def parse(tokens: List[str],
              quoted: bool = False) -> Tuple[SExp, List[str]]:
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

            if quoted:
                list_tail, tokens = read_list_tail(tokens)
                return to_slist(list_tail), tokens[1:]

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

            return parse_call(parsed_first, tokens)
        elif is_number(tokens[0]):
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

    def parse_call(func: SExp,
                   tokens: List[str]) -> Tuple[SExp, List[str]]:
        args, tokens = read_list_tail(tokens)
        return SCall(func, args), tokens[1:]

    def parse_quote(tokens: List[str]) -> Tuple[SExp, List[str]]:
        quoted, tokens = parse(tokens, quoted=True)
        return Quote(quoted), tokens

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
