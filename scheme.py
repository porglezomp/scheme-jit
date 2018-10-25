from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable, Sequence, Union


@dataclass
class SExp:
    """An s-expression base class"""
    pass


@dataclass(frozen=True, order=True)
class SNum(SExp):
    """A lisp number"""
    value: int

    def __str__(self):
        return str(self.value)


@dataclass(frozen=True)
class SSym(SExp):
    """A lisp symbol"""
    name: str

    def __str__(self):
        return self.name


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

    def is_list(self):
        if self.second is Nil:
            return True

        if not isinstance(self.second, SPair):
            return False

        return self.second.is_list()

    def __iter__(self):
        return PairIter(self)

    def __str__(self) -> str:
        if self.is_list():
            return f"({' '.join((str(item) for item in self))})"

        return f"({str(self.first)} . {str(self.second)})"


class PairIter:
    def __init__(self, pair: SPair):
        self._expr: SExp = pair

    def __next__(self):
        if self._expr is Nil:
            raise StopIteration

        if not isinstance(self._expr, SPair):
            val = self._expr
            self._expr = Nil
            return val

        val = self._expr.first
        self._expr = self._expr.second

        return val

    def __iter__(self):
        return self


class NilType(SExp):
    pass


Nil = NilType()

SList = Union[SPair, NilType]


def to_slist(x: Sequence[SExp]) -> SList:
    acc: SList = Nil
    for item in reversed(x):
        acc = SPair(item, acc)

    return acc


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
