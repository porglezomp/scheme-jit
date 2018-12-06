from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

import sexp
from visitor import Visitor


@dataclass(frozen=True)
class SchemeObjectType:
    pass


SchemeObject = SchemeObjectType()


@dataclass(frozen=True)
class SchemeBottomType(SchemeObjectType):
    pass


SchemeBottom = SchemeBottomType()


@dataclass(frozen=True)
class SchemeNumType(SchemeObjectType):
    pass


SchemeNum = SchemeNumType()


@dataclass(frozen=True)
class SchemeBoolType(SchemeObjectType):
    pass


SchemeBool = SchemeBoolType()


@dataclass(frozen=True)
class SchemeSymType(SchemeObjectType):
    pass


SchemeSym = SchemeSymType()


@dataclass(frozen=True)
class SchemeVectType(SchemeObjectType):
    length: Optional[int]


@dataclass(frozen=True)
class SchemeFunctionType(SchemeObjectType):
    arity: Optional[int]


@dataclass(frozen=True)
class SchemeQuotedType(SchemeObjectType):
    expr_type: Type[sexp.SExp]


TypeTuple = Tuple[SchemeObjectType, ...]


@dataclass
class SExpWrapper:
    expr: sexp.SExp

    def __eq__(self, other: Any) -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash(id(self.expr))


class FunctionTypeAnalyzer(Visitor):
    def __init__(self, param_types: Dict[sexp.SSym, SchemeObjectType]):
        self._param_types = param_types
        self._expr_types: Dict[SExpWrapper, SchemeObjectType] = {}

    def visit_SFunction(self, func: sexp.SFunction) -> None:
        if func.is_lambda:
            self._expr_types[SExpWrapper(func)] = (
                SchemeFunctionType(len(func.params)))
            # Lambda bodies will be analyzed separately when they're called
        else:
            super().visit(func.body)

    def visit_SNum(self, num: sexp.SNum) -> None:
        self._expr_types[SExpWrapper(num)] = SchemeNumType()

    def visit_SBool(self, sbool: sexp.SBool) -> None:
        self._expr_types[SExpWrapper(sbool)] = SchemeBoolType()

    def visit_SSym(self, sym: sexp.SSym) -> None:
        if sym in self._param_types:
            self._expr_types[SExpWrapper(sym)] = self._param_types[sym]
        elif sym in _BUILTINS_RETURN_TYPES:
            self._expr_types[SExpWrapper(sym)] = _BUILTINS_RETURN_TYPES[sym]
        elif sym in _BUILTINS_FUNC_TYPES:
            self._expr_types[SExpWrapper(sym)] = _BUILTINS_FUNC_TYPES[sym]
        else:
            self._expr_types[SExpWrapper(sym)] = SchemeSym

    def visit_SVect(self, vect: sexp.SVect) -> None:
        self._expr_types[SExpWrapper(vect)] = SchemeVectType(len(vect.items))

    def visit_Quote(self, quote: sexp.Quote) -> None:
        type_: SchemeObjectType
        if isinstance(quote.expr, sexp.SPair) and quote.expr.is_list():
            # Quoted lists are turned into pairs
            type_ = SchemeVectType(2)
        else:
            type_ = SchemeQuotedType(type(quote.expr))

        self._expr_types[SExpWrapper(quote)] = type_

    def visit_SCall(self, call: sexp.SCall) -> None:
        super().visit_SCall(call)
        assert False, 'fixme: builtin func return types'

    def visit_SConditional(self, cond: sexp.SConditional) -> None:
        super().visit_SConditional(cond)

        then_type = self._expr_types[SExpWrapper(cond.then_expr)]
        else_type = self._expr_types[SExpWrapper(cond.else_expr)]
        if then_type == else_type:
            self._expr_types[SExpWrapper(cond)] = then_type
        else:
            self._expr_types[SExpWrapper(cond)] = SchemeObjectType()


_BUILTINS_RETURN_TYPES: Dict[sexp.SSym, SchemeObjectType] = {
    sexp.SSym('inst/typeof'): SchemeSym,
    sexp.SSym('inst/trap'): SchemeBottom,
    sexp.SSym('inst/trace'): SchemeNum,
    sexp.SSym('inst/breakpoint'): SchemeNum,
    sexp.SSym('inst/alloc'): SchemeVectType(None),
    sexp.SSym('inst/load'): SchemeObject,
    sexp.SSym('inst/store'): SchemeVectType(None),
    sexp.SSym('inst/length'): SchemeNum,

    sexp.SSym('inst/+'): SchemeNum,
    sexp.SSym('inst/-'): SchemeNum,
    sexp.SSym('inst/*'): SchemeNum,
    sexp.SSym('inst//'): SchemeNum,
    sexp.SSym('inst/%'): SchemeNum,
    sexp.SSym('inst/number='): SchemeBool,
    sexp.SSym('inst/symbol='): SchemeBool,
    sexp.SSym('inst/pointer='): SchemeBool,
    sexp.SSym('inst/number<'): SchemeBool,

    sexp.SSym('trap'): SchemeBottom,
    sexp.SSym('trace'): SchemeNum,
    sexp.SSym('breakpoint'): SchemeNum,
    sexp.SSym('assert'): SchemeNum,
    sexp.SSym('typeof'): SchemeSym,
    sexp.SSym('not'): SchemeBool,

    sexp.SSym('number?'): SchemeBool,
    sexp.SSym('symbol?'): SchemeBool,
    sexp.SSym('vector?'): SchemeBool,
    sexp.SSym('function?'): SchemeBool,
    sexp.SSym('bool?'): SchemeBool,
    sexp.SSym('pair?'): SchemeBool,
    sexp.SSym('nil?'): SchemeBool,

    sexp.SSym('+'): SchemeNum,
    sexp.SSym('-'): SchemeNum,
    sexp.SSym('*'): SchemeNum,
    sexp.SSym('/'): SchemeNum,
    sexp.SSym('%'): SchemeNum,

    sexp.SSym('pointer='): SchemeBool,
    sexp.SSym('symbol='): SchemeBool,
    sexp.SSym('number='): SchemeBool,
    sexp.SSym('number<'): SchemeBool,

    sexp.SSym('vector-length'): SchemeNum,
    sexp.SSym('vector-index'): SchemeObject,

    sexp.SSym('vector-set!'): SchemeVectType(None),
    sexp.SSym('vector-make/recur'): SchemeVectType(None),
    sexp.SSym('vector-make'): SchemeVectType(None),
}

_BUILTINS_FUNC_TYPES: Dict[sexp.SSym, SchemeObjectType] = {
    sexp.SSym('inst/typeof'):  SchemeFunctionType(1),
    sexp.SSym('inst/trap'): SchemeFunctionType(0),
    sexp.SSym('inst/trace'): SchemeFunctionType(1),
    sexp.SSym('inst/breakpoint'): SchemeFunctionType(0),
    sexp.SSym('inst/alloc'): SchemeFunctionType(1),
    sexp.SSym('inst/load'): SchemeFunctionType(2),
    sexp.SSym('inst/store'): SchemeFunctionType(3),
    sexp.SSym('inst/length'): SchemeFunctionType(1),

    sexp.SSym('inst/+'): SchemeFunctionType(2),
    sexp.SSym('inst/-'): SchemeFunctionType(2),
    sexp.SSym('inst/*'): SchemeFunctionType(2),
    sexp.SSym('inst//'): SchemeFunctionType(2),
    sexp.SSym('inst/%'): SchemeFunctionType(2),
    sexp.SSym('inst/number='): SchemeFunctionType(2),
    sexp.SSym('inst/symbol='): SchemeFunctionType(2),
    sexp.SSym('inst/pointer='): SchemeFunctionType(2),
    sexp.SSym('inst/number<'): SchemeFunctionType(2),

    sexp.SSym('trap'): SchemeFunctionType(0),
    sexp.SSym('trace'): SchemeFunctionType(1),
    sexp.SSym('breakpoint'): SchemeFunctionType(0),
    sexp.SSym('assert'): SchemeFunctionType(1),
    sexp.SSym('typeof'): SchemeFunctionType(1),
    sexp.SSym('not'): SchemeFunctionType(1),

    sexp.SSym('number?'): SchemeFunctionType(1),
    sexp.SSym('symbol?'): SchemeFunctionType(1),
    sexp.SSym('vector?'): SchemeFunctionType(1),
    sexp.SSym('function?'): SchemeFunctionType(1),
    sexp.SSym('bool?'): SchemeFunctionType(1),
    sexp.SSym('pair?'): SchemeFunctionType(1),
    sexp.SSym('nil?'): SchemeFunctionType(1),

    sexp.SSym('+'): SchemeFunctionType(2),
    sexp.SSym('-'): SchemeFunctionType(2),
    sexp.SSym('*'): SchemeFunctionType(2),
    sexp.SSym('/'): SchemeFunctionType(2),
    sexp.SSym('%'): SchemeFunctionType(2),

    sexp.SSym('pointer='): SchemeFunctionType(2),
    sexp.SSym('symbol='): SchemeFunctionType(2),
    sexp.SSym('number='): SchemeFunctionType(2),
    sexp.SSym('number<'): SchemeFunctionType(2),

    sexp.SSym('vector-length'): SchemeFunctionType(1),
    sexp.SSym('vector-index'): SchemeFunctionType(2),

    sexp.SSym('vector-set!'): SchemeFunctionType(3),
    sexp.SSym('vector-make/recur'): SchemeFunctionType(4),
    sexp.SSym('vector-make'): SchemeFunctionType(2),
}
