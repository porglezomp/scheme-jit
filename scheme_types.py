from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple, Type, cast

import sexp
from visitor import Visitor


@dataclass(frozen=True)
class SchemeObjectType:
    def __lt__(self, other: object) -> bool:
        return issubclass(type(self), type(other))

    def join_with(self, other: object) -> SchemeObjectType:
        assert isinstance(other, SchemeObjectType)
        if self < other:
            return other
        elif other < self:
            return self
        else:
            return SchemeObject


SchemeObject = SchemeObjectType()


@dataclass(frozen=True)
class SchemeBottomType(SchemeObjectType):
    pass


SchemeBottom = SchemeBottomType()


@dataclass(frozen=True)
class SchemeNumType(SchemeObjectType):
    pass
    # value: Optional[int] = None

    # def join_with(self, other: object) -> SchemeObjectType:
    #     if isinstance(other, SchemeNumType):
    #         return self if self.value == other.value else SchemeNum

    #     return super().join_with(other)


SchemeNum = SchemeNumType()


@dataclass(frozen=True)
class SchemeBoolType(SchemeObjectType):
    pass
    # value: Optional[bool] = None

    # def join_with(self, other: object) -> SchemeObjectType:
    #     if isinstance(other, SchemeBoolType):
    #         return self if self.value == other.value else SchemeBool

    #     return super().join_with(other)


SchemeBool = SchemeBoolType()


@dataclass(frozen=True)
class SchemeSymType(SchemeObjectType):
    pass
    # value: Optional[str] = None

    # def join_with(self, other: object) -> SchemeObjectType:
    #     if isinstance(other, SchemeBoolType):
    #         return self if self.value == other.value else SchemeSym

    #     return super().join_with(other)


SchemeSym = SchemeSymType()


@dataclass(frozen=True)
class SchemeVectType(SchemeObjectType):
    length: Optional[int]

    def __lt__(self, other: object) -> bool:
        if isinstance(other, SchemeVectType):
            return other.length is None or self.length == other.length

        return super().__lt__(other)

    def join_with(self, other: object) -> SchemeObjectType:
        if isinstance(other, SchemeVectType):
            length = self.length if self.length == other.length else None
            return SchemeVectType(length)

        return super().join_with(other)


@dataclass(frozen=True)
class SchemeFunctionType(SchemeObjectType):
    arity: Optional[int]
    return_type: SchemeObjectType = SchemeObject

    def __lt__(self, other: object) -> bool:
        if isinstance(other, SchemeFunctionType):
            return (
                (other.arity is None or self.arity == other.arity)
                and self.return_type < other.return_type
            )

        return super().__lt__(other)

    def join_with(self, other: object) -> SchemeObjectType:
        if isinstance(other, SchemeFunctionType):
            arity = self.arity if self.arity == other.arity else None
            return_type = self.return_type.join_with(other.return_type)
            return SchemeFunctionType(arity, return_type)

        return super().join_with(other)


@dataclass(frozen=True)
class SchemeQuotedType(SchemeObjectType):
    expr_type: Type[sexp.SExp]


TypeTuple = Tuple[SchemeObjectType, ...]


@dataclass
class SExpWrapper:
    expr: sexp.SExp

    def __eq__(self, other: object) -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash(id(self.expr))


class FunctionTypeAnalyzer(Visitor):
    def __init__(self, param_types: Mapping[sexp.SSym, SchemeObjectType],
                 global_env: Dict[sexp.SSym, sexp.Value]):
        self._param_types = param_types
        self._expr_types: Dict[SExpWrapper, SchemeObjectType] = {}

        self._global_env = global_env

        self._function_type: Optional[SchemeFunctionType] = None

    def get_function_type(self) -> SchemeFunctionType:
        assert self._function_type is not None
        return self._function_type

    def get_expr_types(self) -> Dict[SExpWrapper, SchemeObjectType]:
        return copy.copy(self._expr_types)

    def get_expr_type(self, expr: sexp.SExp) -> SchemeObjectType:
        return self._expr_types[SExpWrapper(expr)]

    def _set_expr_type(self, expr: sexp.SExp, type_: SchemeObjectType) -> None:
        self._expr_types[SExpWrapper(expr)] = type_

    def expr_type_known(self, expr: sexp.SExp) -> bool:
        return SExpWrapper(expr) in self._expr_types

    def visit_SFunction(self, func: sexp.SFunction) -> None:
        if func.is_lambda:
            self._set_expr_type(func, SchemeFunctionType(len(func.params)))
            # Lambda bodies will be analyzed separately when they're called
        else:
            for param in func.params:
                super().visit(param)
            super().visit(func.body)

            self._function_type = SchemeFunctionType(
                len(func.params),
                self.get_expr_type(list(func.body)[-1])
            )
            self._set_expr_type(func, self._function_type)

    def visit_SNum(self, num: sexp.SNum) -> None:
        self._set_expr_type(num, SchemeNum)

    def visit_SBool(self, sbool: sexp.SBool) -> None:
        self._set_expr_type(sbool, SchemeBool)

    def visit_SSym(self, sym: sexp.SSym) -> None:
        if sym in self._param_types:
            self._set_expr_type(sym, self._param_types[sym])
        elif sym in _BUILTINS_FUNC_TYPES:
            self._set_expr_type(sym, _BUILTINS_FUNC_TYPES[sym])
        elif sym in self._global_env:
            func = self._global_env[sym]
            assert isinstance(func, sexp.SFunction)
            self._set_expr_type(sym, SchemeFunctionType(len(func.params)))
        else:
            self._set_expr_type(sym, SchemeObject)

    def visit_SVect(self, vect: sexp.SVect) -> None:
        self._set_expr_type(vect, SchemeVectType(len(vect.items)))

    def visit_Quote(self, quote: sexp.Quote) -> None:
        type_: SchemeObjectType
        if isinstance(quote.expr, sexp.SPair) and quote.expr.is_list():
            # Quoted lists are turned into pairs
            self._set_expr_type(quote, SchemeVectType(2))
        elif isinstance(quote.expr, sexp.SSym):
            self._set_expr_type(quote, SchemeSym)
        else:
            self._set_expr_type(quote, SchemeQuotedType(type(quote.expr)))

    def visit_SCall(self, call: sexp.SCall) -> None:
        super().visit_SCall(call)
        if call.func == sexp.SSym('vector-make') and len(call.args) == 2:
            size_arg = call.args[0]
            size = size_arg.value if isinstance(size_arg, sexp.SNum) else None
            self._set_expr_type(call, SchemeVectType(size))
        elif self.expr_type_known(call.func):
            func_type = self.get_expr_type(call.func)
            if isinstance(func_type, SchemeFunctionType):
                self._set_expr_type(call, func_type.return_type)
            else:
                self._set_expr_type(call, SchemeObject)
        else:
            self._set_expr_type(call, SchemeObject)

    def visit_SConditional(self, cond: sexp.SConditional) -> None:
        super().visit_SConditional(cond)

        conditional_type = self.get_expr_type(cond.test)
        then_type = self.get_expr_type(cond.then_expr)
        else_type = self.get_expr_type(cond.else_expr)
        if (not isinstance(conditional_type, SchemeBoolType)
                or conditional_type.value is None):
            self._set_expr_type(cond, then_type.join_with(else_type))
        elif conditional_type.value:
            self._set_expr_type(cond, then_type)
        else:
            self._set_expr_type(cond, else_type)


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
