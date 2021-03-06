from __future__ import annotations

import copy
from abc import abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import (Any, Callable, ClassVar, Dict, List, Mapping, Optional,
                    Tuple, Type, cast)

import sexp
from sexp import SSym
from visitor import Visitor


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
    def type_name(self) -> sexp.SSym:
        raise NotImplementedError


@dataclass(frozen=True)
class SchemeNumType(SchemeValueType):
    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('number')

    def __str__(self) -> str:
        return 'number'

    def type_name(self) -> sexp.SSym:
        return sexp.SSym('number')


SchemeNum = SchemeNumType()


@dataclass(frozen=True)
class SchemeBoolType(SchemeValueType):
    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('bool')

    def __str__(self) -> str:
        return 'bool'

    def type_name(self) -> sexp.SSym:
        return sexp.SSym('bool')


SchemeBool = SchemeBoolType()


@dataclass(frozen=True)
class SchemeSymType(SchemeValueType):
    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('symbol')

    def __str__(self) -> str:
        return 'symbol'

    def type_name(self) -> sexp.SSym:
        return sexp.SSym('symbol')


SchemeSym = SchemeSymType()


@dataclass(frozen=True)
class SchemeVectType(SchemeValueType):
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

    def symbol(self) -> Optional[sexp.SSym]:
        return SSym('function')

    def __str__(self) -> str:
        if self.arity is not None:
            return f'function[{self.arity}, {self.return_type}]'
        return 'function[{self.return_type}]'

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


@dataclass
class SExpWrapper:
    expr: sexp.SExp

    def __eq__(self, other: object) -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash(id(self.expr))


def get_type(value: sexp.Value) -> SchemeObjectType:
    visitor = CallArgsTypeAnalyzer()
    visitor.visit(value)
    return visitor.arg_types[0]


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

        # Keep track of whether we've hit a lambda
        self._inside_function = False

    def get_function_type(self) -> SchemeFunctionType:
        assert self._function_type is not None
        return self._function_type

    def get_expr_types(self) -> Dict[SExpWrapper, SchemeObjectType]:
        return copy.copy(self._expr_types)

    def get_expr_type(self, expr: sexp.SExp) -> SchemeObjectType:
        return self._expr_types[SExpWrapper(expr)]

    def set_expr_type(self, expr: sexp.SExp, type_: SchemeObjectType) -> None:
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

    def visit_SBegin(self, begin: sexp.SBegin) -> None:
        assert len(begin.exprs) != 0, 'begin bodies must not be empty'
        super().visit_SBegin(begin)
        self.set_expr_type(begin, self.get_expr_type(begin.exprs[-1]))

    def visit_SFunction(self, func: sexp.SFunction) -> None:
        if self._inside_function:
            self.set_expr_type(func, SchemeFunctionType(len(func.params)))
            # Lambda bodies will be analyzed separately when they're called
        else:
            self._inside_function = True
            for param in func.params:
                super().visit(param)
            super().visit(func.body)

            self._function_type = SchemeFunctionType(
                len(func.params),
                self.get_expr_type(list(func.body)[-1])
            )
            self.set_expr_type(func, self._function_type)

    def visit_SNum(self, num: sexp.SNum) -> None:
        self.set_expr_type(num, SchemeNum)
        self.set_expr_value(num, num)

    def visit_SBool(self, sbool: sexp.SBool) -> None:
        self.set_expr_type(sbool, SchemeBool)
        self.set_expr_value(sbool, sbool)

    def visit_SSym(self, sym: sexp.SSym) -> None:
        if sym in self._param_types:
            self.set_expr_type(sym, self._param_types[sym])
        elif sym in _BUILTINS_FUNC_TYPES:
            self.set_expr_type(sym, _BUILTINS_FUNC_TYPES[sym])
        elif sym in self._global_env:
            func = self._global_env[sym]
            assert isinstance(func, sexp.SFunction)
            self.set_expr_type(sym, SchemeFunctionType(len(func.params)))
        else:
            self.set_expr_type(sym, SchemeObject)

    def visit_SVect(self, vect: sexp.SVect) -> None:
        super().visit_SVect(vect)
        self.set_expr_type(vect, SchemeVectType(len(vect.items)))

    def visit_Quote(self, quote: sexp.Quote) -> None:
        type_: SchemeObjectType
        if isinstance(quote.expr, sexp.SPair) and quote.expr.is_list():
            # Quoted lists are turned into pairs
            self.set_expr_type(quote, SchemeVectType(2))
        elif isinstance(quote.expr, sexp.SSym):
            self.set_expr_type(quote, SchemeSym)
        else:
            self.set_expr_type(quote, SchemeQuotedType(type(quote.expr)))

    def visit_SCall(self, call: sexp.SCall) -> None:
        super().visit_SCall(call)
        if (isinstance(call.func, sexp.SSym)
                and call.func in _builtin_const_exprs):
            _builtin_const_exprs[call.func](self).eval_expr(call)
        elif self.expr_type_known(call.func):
            func_type = self.get_expr_type(call.func)
            if isinstance(func_type, SchemeFunctionType):
                self.set_expr_type(call, func_type.return_type)
            else:
                self.set_expr_type(call, SchemeObject)
        else:
            self.set_expr_type(call, SchemeObject)

    def visit_SConditional(self, cond: sexp.SConditional) -> None:
        super().visit_SConditional(cond)

        then_type = self.get_expr_type(cond.then_expr)
        else_type = self.get_expr_type(cond.else_expr)

        if (self.get_expr_type(cond.test) == SchemeBool
                and self.expr_value_known(cond.test)):
            expr_val = self.get_expr_value(cond.test)
            assert isinstance(expr_val, sexp.SBool)
            if expr_val.value:
                self.set_expr_type(cond, then_type)
            else:
                self.set_expr_type(cond, else_type)
        else:
            self.set_expr_type(cond, then_type.join_with(else_type))


_BUILTINS_FUNC_TYPES: Dict[sexp.SSym, SchemeObjectType] = {
    sexp.SSym('inst/typeof'):  SchemeFunctionType(1, SchemeSym),
    sexp.SSym('inst/trap'): SchemeFunctionType(0, SchemeBottom),
    sexp.SSym('inst/trace'): SchemeFunctionType(1, SchemeNum),
    sexp.SSym('inst/display'): SchemeFunctionType(1, SchemeNum),
    sexp.SSym('inst/newline'): SchemeFunctionType(0, SchemeNum),
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
    sexp.SSym('display'): SchemeFunctionType(1, SchemeNum),
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

    sexp.SSym('list'): SchemeFunctionType(1, SchemeVectType(None)),

    sexp.SSym('cons'): SchemeVectType(2)
}


class BuiltinCallTypeEvaler:
    expected_arg_types: ClassVar[Tuple[SchemeObjectType, ...]]
    base_return_type: ClassVar[SchemeObjectType]

    def __init__(self, expr_types: FunctionTypeAnalyzer):
        self.expr_types = expr_types

    def eval_expr(self, call: sexp.SCall) -> None:
        self.expr_types.set_expr_type(call, self.base_return_type)

        if len(call.args) != len(self.expected_arg_types):
            return

        call_args: List[CallArg] = []
        for arg, expected_type in zip(call.args, self.expected_arg_types):
            arg_type = self.expr_types.get_expr_type(arg)
            if not (arg_type < expected_type):
                return

            arg_value = (self.expr_types.get_expr_value(arg)
                         if self.expr_types.expr_value_known(arg) else None)

            call_args.append(CallArg(arg_type, arg_value))

        self._eval_expr_impl(call, *call_args)

    def _eval_expr_impl(self, call: sexp.SCall, *args: CallArg) -> None:
        pass


@dataclass(eq=False)
class CallArg:
    type_: SchemeObjectType
    value: Optional[sexp.Value]


_DecoratorType = Callable[[Type[BuiltinCallTypeEvaler]],
                          Type[BuiltinCallTypeEvaler]]


def _register_const_call_expr(func_name: str) -> _DecoratorType:
    def decorator(cls: Type[BuiltinCallTypeEvaler]
                  ) -> Type[BuiltinCallTypeEvaler]:
        _builtin_const_exprs[sexp.SSym(func_name)] = cls
        return cls

    return decorator


_builtin_const_exprs: Dict[sexp.SSym, Type[BuiltinCallTypeEvaler]] = {}


@_register_const_call_expr('typeof')
class Typeof(BuiltinCallTypeEvaler):
    expected_arg_types = (SchemeValueType(),)
    base_return_type = SchemeSym

    def _eval_expr_impl(self, call: sexp.SCall, *args: CallArg) -> None:
        [arg] = args
        if isinstance(arg.type_, SchemeValueType):
            self.expr_types.set_expr_value(call, arg.type_.type_name())


@_register_const_call_expr('not')
class Not(BuiltinCallTypeEvaler):
    expected_arg_types = (SchemeBool,)
    base_return_type = SchemeBool

    def _eval_expr_impl(self, call: sexp.SCall, *args: CallArg) -> None:
        [arg] = args
        if isinstance(arg.value, sexp.SBool):
            self.expr_types.set_expr_value(
                call, sexp.SBool(not arg.value.value))


class TypeQuery(BuiltinCallTypeEvaler):
    expected_arg_types = (SchemeObject,)
    base_return_type = SchemeBool

    query_type: SchemeObjectType

    def _eval_expr_impl(self, call: sexp.SCall, *args: CallArg) -> None:
        [type_query_arg] = args

        # The type being SchemeObject probably indicates that
        # we don't know the type, so we don't know for sure if the
        # query is false.
        if type_query_arg.type_ == SchemeObject:
            return

        self.expr_types.set_expr_value(
            call,
            sexp.SBool(type_query_arg.type_ < self.query_type))


@_register_const_call_expr('number?')
class IsNumber(TypeQuery):
    query_type = SchemeNum


@_register_const_call_expr('symbol?')
class IsSymbol(TypeQuery):
    query_type = SchemeSym


@_register_const_call_expr('vector?')
class IsVector(TypeQuery):
    query_type = SchemeVectType(None)


@_register_const_call_expr('function?')
class IsFunction(TypeQuery):
    query_type = SchemeFunctionType(None)


@_register_const_call_expr('bool?')
class IsBool(TypeQuery):
    query_type = SchemeBool


@_register_const_call_expr('symbol=')
class SymbolEq(BuiltinCallTypeEvaler):
    expected_arg_types = (SchemeSym, SchemeSym)
    base_return_type = SchemeBool

    def _eval_expr_impl(self, call: sexp.SCall, *args: CallArg) -> None:
        [first, second] = args

        if first.value is not None and second.value is not None:
            self.expr_types.set_expr_value(
                call,
                sexp.SBool(first.value == second.value))


@_register_const_call_expr('vector-make')
class VectorMake(BuiltinCallTypeEvaler):
    expected_arg_types = (SchemeNum, SchemeObject)
    base_return_type = SchemeVectType(None)

    def _eval_expr_impl(self, call: sexp.SCall, *args: CallArg) -> None:
        [size_arg, _] = args
        size = (size_arg.value.value
                if isinstance(size_arg.value, sexp.SNum) else None)
        self.expr_types.set_expr_type(call, SchemeVectType(size))
