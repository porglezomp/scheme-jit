import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Type

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
    return_type: SchemeObjectType = SchemeObject


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
    def __init__(self, param_types: Mapping[sexp.SSym, SchemeObjectType]):
        self._param_types = param_types
        self._expr_types: Dict[SExpWrapper, SchemeObjectType] = {}

        self._function_type: Optional[SchemeFunctionType] = None

    def get_function_type(self) -> SchemeFunctionType:
        assert self._function_type is not None
        return self._function_type

    def get_expr_types(self) -> Dict[SExpWrapper, SchemeObjectType]:
        return copy.copy(self._expr_types)

    def _get_expr_type(self, expr: sexp.SExp) -> SchemeObjectType:
        return self._expr_types[SExpWrapper(expr)]

    def _set_expr_type(self, expr: sexp.SExp, type_: SchemeObjectType) -> None:
        self._expr_types[SExpWrapper(expr)] = type_

    def _expr_type_known(self, expr: sexp.SExp) -> bool:
        return SExpWrapper(expr) in self._expr_types

    def visit_SFunction(self, func: sexp.SFunction) -> None:
        if func.is_lambda:
            self._set_expr_type(func, SchemeFunctionType(len(func.params)))
            # Lambda bodies will be analyzed separately when they're called
        else:
            super().visit(func.body)

            self._function_type = SchemeFunctionType(
                len(func.params),
                self._get_expr_type(list(func.body)[-1])
            )
            self._set_expr_type(func, self._function_type)

    def visit_SNum(self, num: sexp.SNum) -> None:
        self._set_expr_type(num, SchemeNumType())

    def visit_SBool(self, sbool: sexp.SBool) -> None:
        self._set_expr_type(sbool, SchemeBoolType())

    def visit_SSym(self, sym: sexp.SSym) -> None:
        if sym in self._param_types:
            self._set_expr_type(sym, self._param_types[sym])
        elif sym in _BUILTINS_FUNC_TYPES:
            self._set_expr_type(sym, _BUILTINS_FUNC_TYPES[sym])
        else:
            self._set_expr_type(sym, SchemeObject)

    def visit_SVect(self, vect: sexp.SVect) -> None:
        self._set_expr_type(vect, SchemeVectType(len(vect.items)))

    def visit_Quote(self, quote: sexp.Quote) -> None:
        type_: SchemeObjectType
        if isinstance(quote.expr, sexp.SPair) and quote.expr.is_list():
            # Quoted lists are turned into pairs
            type_ = SchemeVectType(2)
        else:
            type_ = SchemeQuotedType(type(quote.expr))

        self._set_expr_type(quote, type_)

    def visit_SCall(self, call: sexp.SCall) -> None:
        super().visit_SCall(call)
        if self._expr_type_known(call.func):
            func_type = self._get_expr_type(call.func)
            if isinstance(func_type, SchemeFunctionType):
                self._set_expr_type(call, func_type.return_type)
            else:
                self._set_expr_type(call, SchemeObject)
        else:
            self._set_expr_type(call, SchemeObject)

    def visit_SConditional(self, cond: sexp.SConditional) -> None:
        super().visit_SConditional(cond)

        then_type = self._get_expr_type(cond.then_expr)
        else_type = self._get_expr_type(cond.else_expr)
        if then_type == else_type:
            self._set_expr_type(cond, then_type)
        else:
            self._set_expr_type(cond, SchemeObjectType())


_BUILTINS_FUNC_TYPES: Dict[sexp.SSym, SchemeObjectType] = {
    sexp.SSym('inst/typeof'):  SchemeFunctionType(1, SchemeSym),
    sexp.SSym('inst/trap'): SchemeFunctionType(0, SchemeBottom),
    sexp.SSym('inst/trace'): SchemeFunctionType(1, SchemeNum),
    sexp.SSym('inst/breakpoint'): SchemeFunctionType(0, SchemeNum),
    sexp.SSym('inst/alloc'): SchemeFunctionType(1, SchemeVectType(None)),
    sexp.SSym('inst/load'): SchemeFunctionType(2, SchemeObject),
    sexp.SSym('inst/store'): SchemeFunctionType(3, SchemeVectType(None)),
    sexp.SSym('inst/length'): SchemeFunctionType(1, SchemeNum),

    sexp.SSym('inst/+'): SchemeFunctionType(2, SchemeNum),
    sexp.SSym('inst/-'): SchemeFunctionType(2, SchemeNum),
    sexp.SSym('inst/*'): SchemeFunctionType(2, SchemeNum),
    sexp.SSym('inst//'): SchemeFunctionType(2, SchemeNum),
    sexp.SSym('inst/%'): SchemeFunctionType(2, SchemeNum),
    sexp.SSym('inst/number='): SchemeFunctionType(2, SchemeBool),
    sexp.SSym('inst/symbol='): SchemeFunctionType(2, SchemeBool),
    sexp.SSym('inst/pointer='): SchemeFunctionType(2, SchemeBool),
    sexp.SSym('inst/number<'): SchemeFunctionType(2, SchemeBool),

    sexp.SSym('trap'): SchemeFunctionType(0, SchemeBottom),
    sexp.SSym('trace'): SchemeFunctionType(1, SchemeNum),
    sexp.SSym('breakpoint'): SchemeFunctionType(0, SchemeNum),
    sexp.SSym('assert'): SchemeFunctionType(1, SchemeNum),
    sexp.SSym('typeof'): SchemeFunctionType(1, SchemeSym),
    sexp.SSym('not'): SchemeFunctionType(1, SchemeBool),

    sexp.SSym('number?'): SchemeFunctionType(1, SchemeBool),
    sexp.SSym('symbol?'): SchemeFunctionType(1, SchemeBool),
    sexp.SSym('vector?'): SchemeFunctionType(1, SchemeBool),
    sexp.SSym('function?'): SchemeFunctionType(1, SchemeBool),
    sexp.SSym('bool?'): SchemeFunctionType(1, SchemeBool),
    sexp.SSym('pair?'): SchemeFunctionType(1, SchemeBool),
    sexp.SSym('nil?'): SchemeFunctionType(1, SchemeBool),

    sexp.SSym('+'): SchemeFunctionType(2, SchemeNum),
    sexp.SSym('-'): SchemeFunctionType(2, SchemeNum),
    sexp.SSym('*'): SchemeFunctionType(2, SchemeNum),
    sexp.SSym('/'): SchemeFunctionType(2, SchemeNum),
    sexp.SSym('%'): SchemeFunctionType(2, SchemeNum),

    sexp.SSym('pointer='): SchemeFunctionType(2, SchemeBool),
    sexp.SSym('symbol='): SchemeFunctionType(2, SchemeBool),
    sexp.SSym('number='): SchemeFunctionType(2, SchemeBool),
    sexp.SSym('number<'): SchemeFunctionType(2, SchemeBool),

    sexp.SSym('vector-length'): SchemeFunctionType(1, SchemeNum),
    sexp.SSym('vector-index'): SchemeFunctionType(2, SchemeObject),

    sexp.SSym('vector-set!'): SchemeFunctionType(3, SchemeVectType(None)),
    sexp.SSym('vector-make/recur'): (
        SchemeFunctionType(4, SchemeVectType(None))),
    sexp.SSym('vector-make'): SchemeFunctionType(2, SchemeVectType(None)),
}
