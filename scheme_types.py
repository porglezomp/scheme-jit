from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

import sexp
from sexp import SSym


@dataclass(frozen=True)
class SchemeObjectType:
    def symbol(self) -> Optional[sexp.SSym]:
        return None


SchemeObject = SchemeObjectType()


@dataclass(frozen=True)
class SchemeBottomType:
    pass


SchemeBottom = SchemeBottomType()


@dataclass(frozen=True)
class SchemeNumType(SchemeObjectType):
    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('number')


SchemeNum = SchemeNumType()


@dataclass(frozen=True)
class SchemeBoolType(SchemeObjectType):
    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('bool')


SchemeBool = SchemeBoolType()


@dataclass(frozen=True)
class SchemeSymType(SchemeObjectType):
    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('symbol')


SchemeSym = SchemeSymType()


@dataclass(frozen=True)
class SchemeVectType(SchemeObjectType):
    length: Optional[int]

    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('vector')


@dataclass(frozen=True)
class SchemeFunctionType(SchemeObjectType):
    arity: Optional[int]

    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('function')


@dataclass(frozen=True)
class SchemeQuotedType(SchemeObjectType):
    expr_type: Type[sexp.SExp]


TypeTuple = Tuple[SchemeObjectType, ...]
