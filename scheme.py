from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable


@dataclass
class SExp:
    """An s-expression base class"""
    pass


@dataclass(frozen=True)
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
    """An n element vector"""
    items: List[SExp]

    def to_list_items(self) -> Optional[List[SExp]]:
        if len(self.items) == 0:
            return []
        if len(self.items) != 2:
            return None
        rest_items = self.items[1].to_list_items()
        if rest_items is None:
            return None
        return [self.items[0]] + rest_items

    def __str__(self) -> str:
        items = self.to_list_items()
        if items is not None:
            return f"({' '.join(str(i) for i in items)})"
        return f"[{' '.join(str(i) for i in self.items)}]"


def to_slist(x: Iterable[SExp]) -> SExp:
    acc = SVect([])
    for item in reversed(x):
        acc = SVect([item, acc])
    return acc


def parse(x: str) -> List[SExp]:
    tokens = (
        x
        .replace('(', ' ( ')
        .replace(')', ' ) ')
        .replace('[', ' [ ')
        .replace(']', ' ] ')
        .split()
    )

    def parse(tokens: List[str]) -> Tuple[SExp, List[str]]:
        if not tokens:
            raise "Parse Error"
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
