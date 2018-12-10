from __future__ import annotations

from abc import abstractmethod
import copy
from dataclasses import InitVar, dataclass, field
from typing import (Callable, ClassVar, Dict, List, Mapping, Optional, Tuple,
                    Type, cast)

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


class SchemeValueType(SchemeObjectType):
    # @abstractmethod
    def type_name(self) -> sexp.SSym:
        raise NotImplementedError
        # return sexp.SSym('pointer')


@dataclass(frozen=True)
class SchemeNumType(SchemeValueType):
    def type_name(self) -> sexp.SSym:
        return sexp.SSym('number')


SchemeNum = SchemeNumType()


@dataclass(frozen=True)
class SchemeBoolType(SchemeValueType):
    def type_name(self) -> sexp.SSym:
        return sexp.SSym('bool')


SchemeBool = SchemeBoolType()


@dataclass(frozen=True)
class SchemeSymType(SchemeValueType):
    def type_name(self) -> sexp.SSym:
        return sexp.SSym('symbol')

    pass


SchemeSym = SchemeSymType()


@dataclass(frozen=True)
class SchemeVectType(SchemeValueType):
    length: Optional[int]

    def __init__(self, length: Optional[int]):
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        object.__setattr__(
            self,
            'length',
            None if length is not None and length > 4 else length)

    def type_name(self) -> sexp.SSym:
        return sexp.SSym('vector')

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
class SchemeFunctionType(SchemeValueType):
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

    def type_name(self) -> sexp.SSym:
        return sexp.SSym('function')

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


class CallArgsTypeAnalyzer(Visitor):
    def __init__(self) -> None:
        self.arg_types: List[SchemeObjectType] = []

    def visit_SFunction(self, func: sexp.SFunction) -> None:
        self.arg_types.append(SchemeFunctionType(len(func.params)))

    def visit_SNum(self, num: sexp.SNum) -> None:
        self.arg_types.append(SchemeNum)

    def visit_SBool(self, sbool: sexp.SBool) -> None:
        self.arg_types.append(SchemeBool)

    def visit_SSym(self, sym: sexp.SSym) -> None:
        self.arg_types.append(SchemeSym)

    def visit_SVect(self, vect: sexp.SVect) -> None:
        self.arg_types.append(SchemeVectType(len(vect.items)))


class FunctionTypeAnalyzer(Visitor):
    def __init__(self, param_types: Mapping[sexp.SSym, SchemeObjectType],
                 global_env: Dict[sexp.SSym, sexp.Value]):
        self._param_types = param_types
        self._expr_types: Dict[SExpWrapper, SchemeObjectType] = {}
        self._expr_values: Dict[SExpWrapper, sexp.Value] = {}

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

    def get_expr_values(self) -> Dict[SExpWrapper, sexp.Value]:
        return copy.copy(self._expr_values)

    def get_expr_value(self, expr: sexp.SExp) -> sexp.Value:
        return self._expr_values[SExpWrapper(expr)]

    def expr_value_known(self, expr: sexp.SExp) -> bool:
        return SExpWrapper(expr) in self._expr_values

    def set_expr_value(self, expr: sexp.SExp, value: sexp.Value) -> None:
        self._expr_values[SExpWrapper(expr)] = value

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
        self.set_expr_value(num, num)

    def visit_SBool(self, sbool: sexp.SBool) -> None:
        self._set_expr_type(sbool, SchemeBool)
        self.set_expr_value(sbool, sbool)

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
        type_query_val = self._get_type_query_value(call)
        if type_query_val is not None:
            self._set_expr_type(call, SchemeBool)
            self.set_expr_value(call, sexp.SBool(type_query_val))
        elif (isinstance(call.func, sexp.SSym)
                and call.func in _builtin_const_exprs):
            _builtin_const_exprs[call.func](self).eval_expr(call)
        elif call.func == sexp.SSym('vector-make') and len(call.args) == 2:
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

        then_type = self.get_expr_type(cond.then_expr)
        else_type = self.get_expr_type(cond.else_expr)

        if (self.get_expr_type(cond.test) == SchemeBool
                and self.expr_value_known(cond.test)):
            expr_val = self.get_expr_value(cond.test)
            assert isinstance(expr_val, sexp.SBool)
            if expr_val.value:
                self._set_expr_type(cond, then_type)
            else:
                self._set_expr_type(cond, else_type)
        else:
            self._set_expr_type(cond, then_type.join_with(else_type))

    def _get_type_query_value(self, query: sexp.SCall) -> Optional[bool]:
        if self._expr_types is None:
            return None

        func_name = (query.func.name if isinstance(query.func, sexp.SFunction)
                     else query.func)
        if (not isinstance(func_name, sexp.SSym)
                or func_name not in self._TYPE_QUERIES):
            return None

        if len(query.args) != 1:
            return None

        type_query_arg = query.args[0]
        if not self.expr_type_known(type_query_arg):
            return None

        # The type being SchemeObject probably indicates that
        # we don't know the type, so we don't know for sure if the
        # query is false.
        if self.get_expr_type(type_query_arg) == SchemeObject:
            return None

        return (self.get_expr_type(type_query_arg)
                < self._TYPE_QUERIES[cast(sexp.SSym, query.func)])

    _TYPE_QUERIES: Dict[sexp.SSym, SchemeObjectType] = {
        sexp.SSym('number?'): SchemeNum,
        sexp.SSym('symbol?'): SchemeSym,
        sexp.SSym('vector?'): SchemeVectType(None),
        sexp.SSym('function?'): SchemeFunctionType(None),
        sexp.SSym('bool?'): SchemeBool,
        sexp.SSym('pair?'): SchemeVectType(2),
        sexp.SSym('nil?'): SchemeVectType(0),
    }


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

    sexp.SSym('cons'): SchemeVectType(2)
}


@dataclass(eq=False)
class ConstCallExprEvaler:
    expr_types: FunctionTypeAnalyzer
    expected_arg_types: ClassVar[Tuple[SchemeObjectType]]
    require_known_values: ClassVar[bool]

    def eval_expr(self, call: sexp.SCall) -> None:
        if len(call.args) != len(self.expected_arg_types):
            return

        call_args: List[CallArg] = []
        for arg, expected_type in zip(call.args, self.expected_arg_types):
            arg_type = self.expr_types.get_expr_type(arg)
            if not (arg_type < expected_type):
                return

            arg_value = (self.expr_types.get_expr_value(arg)
                         if self.expr_types.expr_value_known(arg) else None)
            if self.require_known_values and arg_value is None:
                return

            call_args.append(CallArg(arg_type, arg_value))

        self._eval_expr_impl(call, *call_args)

    @abstractmethod
    def _eval_expr_impl(self, call: sexp.SCall, *args: CallArg) -> None:
        ...


@dataclass(eq=False)
class CallArg:
    type_: SchemeObjectType
    value: Optional[sexp.Value]


_DecoratorType = Callable[[Type[ConstCallExprEvaler]],
                          Type[ConstCallExprEvaler]]


def _register_const_call_expr(func_name: str) -> _DecoratorType:
    def decorator(cls: Type[ConstCallExprEvaler]) -> Type[ConstCallExprEvaler]:
        _builtin_const_exprs[sexp.SSym(func_name)] = cls
        return cls

    return decorator


_builtin_const_exprs: Dict[sexp.SSym, Type[ConstCallExprEvaler]] = {}


@_register_const_call_expr('typeof')
class Typeof(ConstCallExprEvaler):
    expected_arg_types = (SchemeValueType(),)
    require_known_values = False

    def _eval_expr_impl(self, call: sexp.SCall, *args: CallArg) -> None:
        [arg] = args
        if isinstance(arg.type_, SchemeValueType):
            self.expr_types.set_expr_value(call, arg.type_.type_name())
