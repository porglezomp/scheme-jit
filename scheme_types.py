from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

import sexp
from sexp import SSym


@dataclass(frozen=True)
class SchemeObjectType:
    def symbol(self) -> Optional[sexp.SSym]:
        return None

    def __str__(self) -> str:
        return 'object'

    def join(self, other: SchemeObjectType) -> SchemeObjectType:
        if self == other:
            return self
        return SchemeObject


SchemeObject = SchemeObjectType()


@dataclass(frozen=True)
class SchemeBottomType:
    pass


SchemeBottom = SchemeBottomType()


@dataclass(frozen=True)
class SchemeNumType(SchemeObjectType):
    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('number')

    def __str__(self) -> str:
        return 'number'


SchemeNum = SchemeNumType()


@dataclass(frozen=True)
class SchemeBoolType(SchemeObjectType):
    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('bool')

    def __str__(self) -> str:
        return 'bool'


SchemeBool = SchemeBoolType()


@dataclass(frozen=True)
class SchemeSymType(SchemeObjectType):
    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('symbol')

    def __str__(self) -> str:
        return 'symbol'


SchemeSym = SchemeSymType()


@dataclass(frozen=True)
class SchemeVectType(SchemeObjectType):
    length: Optional[int]

    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('vector')

    def __str__(self) -> str:
        if self.length is not None:
            return f'vector[{self.length}]'
        return 'vector'

    def join(self, other: SchemeObjectType) -> SchemeObjectType:
        if isinstance(other, SchemeVectType):
            if self.length == other.length:
                return SchemeVectType(self.length)
            return SchemeVectType(None)
        return SchemeObject


@dataclass(frozen=True)
class SchemeFunctionType(SchemeObjectType):
    arity: Optional[int]

    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('function')

    def __str__(self) -> str:
        if self.arity is not None:
            return f'function[{self.arity}]'
        return 'function'

    def join(self, other: SchemeObjectType) -> SchemeObjectType:
        if isinstance(other, SchemeFunctionType):
            if self.arity == other.arity:
                return SchemeFunctionType(self.arity)
            return SchemeFunctionType(None)
        return SchemeObject


@dataclass(frozen=True)
class SchemeQuotedType(SchemeObjectType):
    expr_type: Type[sexp.SExp]


TypeTuple = Tuple[SchemeObjectType, ...]
