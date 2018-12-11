from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue
from typing import (Any, Counter, Dict, Generator, Generic, Iterable, Iterator,
                    List, Optional, Set, TypeVar)

import scheme_types
import sexp
from errors import Trap
from scheme_types import SchemeFunctionType, SchemeObjectType, TypeTuple
from sexp import SBool, SExp, SNum, SSym, SVect, Value


@dataclass
class TypeMap:
    types: Dict[Var, SchemeObjectType] = field(default_factory=dict)

    def __getitem__(self, key: Parameter) -> SchemeObjectType:
        try:
            return key.lookup_self_type(self.types)
        except KeyError:
            return scheme_types.SchemeObject

    def __setitem__(self, key: Var, value: SchemeObjectType) -> None:
        self.types[key] = value

    def __repr__(self) -> str:
        parts = ', '.join(f"{k.name}: {v}" for k, v in self.types.items())
        return f"TypeMap({{{parts}}})"

    def join(self, other: TypeMap) -> TypeMap:
        result = TypeMap()
        for key in self.types:
            if key in other.types:
                result[key] = self[key].join(other[key])
        return result


@dataclass
class ValueMap:
    values: Dict[Var, Value] = field(default_factory=dict)

    def __getitem__(self, key: Parameter) -> Optional[Value]:
        try:
            return key.lookup_self(self.values)
        except KeyError:
            return None

    def __setitem__(self, key: Var, value: Optional[Value]) -> None:
        if value is None:
            self.values.pop(key, None)
        else:
            self.values[key] = value

    def __repr__(self) -> str:
        parts = ', '.join(f"{k.name}: {v}" for k, v in self.values.items())
        return f"ValueMap({{{parts}}})"

    def join(self, other: ValueMap) -> ValueMap:
        result = ValueMap()
        for key in self.values:
            if key in other.values:
                if self[key] == other[key]:
                    result[key] = self[key]
        return result

    def get_param(
            self, key: Parameter, allow_func: bool = False,
    ) ->Parameter:
        value = self[key]
        if value is None:
            return key
        param = value.to_param()
        if param is None:
            return key
        if not allow_func and isinstance(param, FuncLit):
            return key
        return param


class Parameter(ABC):
    @abstractmethod
    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        ...

    @abstractmethod
    def lookup_self_type(
            self, env: Dict[Var, SchemeObjectType]) -> SchemeObjectType:
        ...

    @abstractmethod
    def freshen(self, prefix: str) -> Parameter:
        ...


class Inst(ABC):
    @abstractmethod
    def run(self, env: EvalEnv) -> Optional[BB]:
        ...

    @abstractmethod
    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        ...

    def successors(self) -> Iterable[BasicBlock]:
        return []

    @abstractmethod
    def freshen(self, prefix: str) -> None:
        ...

    @abstractmethod
    def constant_fold(self, values: ValueMap) -> Inst:
        ...

    @abstractmethod
    def dests(self) -> List[Var]:
        ...

    @abstractmethod
    def params(self) -> List[Parameter]:
        ...

    @abstractmethod
    def pure(self) -> bool:
        ...


class BB(ABC):
    name: str

    def successors(self) -> Iterable[BasicBlock]:
        return []

    def format_stats(self, stats: Stats) -> str:
        raise NotImplementedError("format_stats")


@dataclass(frozen=True)
class Var(Parameter):
    name: str

    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        return env[self]

    def lookup_self_type(
            self, env: Dict[Var, SchemeObjectType]) -> SchemeObjectType:
        return env[self]

    def __str__(self) -> str:
        return self.name

    def freshen(self, prefix: str) -> Var:
        return Var(f"{prefix}@{self.name}")


@dataclass(frozen=True)
class NumLit(Parameter):
    value: SNum

    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        return self.value

    def lookup_self_type(
            self, env: Dict[Var, SchemeObjectType]) -> SchemeObjectType:
        return scheme_types.SchemeNum

    def __str__(self) -> str:
        return str(self.value)

    def freshen(self, prefix: str) -> NumLit:
        return self


@dataclass(frozen=True)
class SymLit(Parameter):
    value: SSym

    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        return self.value

    def lookup_self_type(
            self, env: Dict[Var, SchemeObjectType]) -> SchemeObjectType:
        return scheme_types.SchemeSym

    def __str__(self) -> str:
        return f"'{self.value}"

    def freshen(self, prefix: str) -> SymLit:
        return self


@dataclass(frozen=True)
class BoolLit(Parameter):
    value: SBool

    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        return self.value

    def lookup_self_type(
            self, env: Dict[Var, SchemeObjectType]) -> SchemeObjectType:
        return scheme_types.SchemeBool

    def __str__(self) -> str:
        return f"{self.value}"

    def freshen(self, prefix: str) -> BoolLit:
        return self


@dataclass(frozen=True)
class FuncLit(Parameter):
    func: sexp.SFunction

    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        return self.func

    def lookup_self_type(
            self, env: Dict[Var, SchemeObjectType]) -> SchemeObjectType:
        return scheme_types.SchemeFunctionType(len(self.func.params))

    def __str__(self) -> str:
        return f"{self.func}"

    def freshen(self, prefix: str) -> FuncLit:
        return self


@dataclass
class Stats:
    inst_type_count: Counter[type] = field(default_factory=Counter)
    block_count: Counter[int] = field(default_factory=Counter)
    inst_count: Counter[int] = field(default_factory=Counter)
    taken_count: Counter[int] = field(default_factory=Counter)
    function_count: Counter[int] = field(default_factory=Counter)
    specialization_dispatch: Counter[int] = field(default_factory=Counter)


class EvalEnv:
    _local_env: Dict[Var, Value]
    _global_env: Dict[SSym, Value]
    stats: Stats

    def __init__(self,
                 local_env: Optional[Dict[Var, Value]] = None,
                 global_env: Optional[Dict[SSym, Value]] = None):
        if local_env is None:
            self._local_env = {}
        else:
            self._local_env = local_env
        if global_env is None:
            self._global_env = {}
        else:
            self._global_env = global_env
        self.stats = Stats()

    def copy(self) -> EvalEnv:
        """Return a shallow copy of the environment."""
        env = EvalEnv(self._local_env.copy(), self._global_env)
        env.stats = self.stats
        return env

    def __getitem__(self, key: Parameter) -> Value:
        """
        Looks up a value in the local environment.

        If the value is not a variable, it returns its runtime value.

        >>> env = EvalEnv()
        >>> env[Var("x0")] = SSym("nil?")
        >>> env[Var("x0")]
        SSym(name='nil?')
        >>> env[NumLit(SNum(42))]
        SNum(value=42)
        """
        return key.lookup_self(self._local_env)

    def __setitem__(self, key: Var, value: Value) -> None:
        self._local_env[key] = value

    def __contains__(self, key: Parameter) -> bool:
        """
        Returns whether the given key is in the local environment.

        Values that aren't variables are always available in the environment.

        >>> env = EvalEnv()
        >>> env[Var("x0")] = SSym("nil?")
        >>> Var("x0") in env
        True
        >>> Var("nil?") in env
        False
        >>> SymLit(SSym("nil?")) in env
        True
        """
        if isinstance(key, Var):
            return key in self._local_env
        return True

    def __repr__(self) -> str:
        return f"EvalEnv({self._local_env})"


class Binop(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NUM_EQ = auto()
    SYM_EQ = auto()
    PTR_EQ = auto()
    NUM_LT = auto()


@dataclass
class BinopInst(Inst):
    dest: Var
    op: Binop
    lhs: Parameter
    rhs: Parameter

    def run(self, env: EvalEnv) -> None:
        lhs = env[self.lhs]
        rhs = env[self.rhs]
        if self.op == Binop.SYM_EQ:
            assert isinstance(lhs, SSym) and isinstance(rhs, SSym)
            env[self.dest] = SBool(lhs == rhs)
        elif self.op == Binop.PTR_EQ:
            env[self.dest] = SBool(lhs.address() == rhs.address())
        else:
            assert isinstance(lhs, SNum) and isinstance(rhs, SNum)
            if self.op == Binop.ADD:
                env[self.dest] = SNum(lhs.value + rhs.value)
            elif self.op == Binop.SUB:
                env[self.dest] = SNum(lhs.value - rhs.value)
            elif self.op == Binop.MUL:
                env[self.dest] = SNum(lhs.value * rhs.value)
            elif self.op == Binop.DIV:
                assert rhs.value != 0
                env[self.dest] = SNum(lhs.value // rhs.value)
            elif self.op == Binop.MOD:
                assert rhs.value != 0
                env[self.dest] = SNum(lhs.value % rhs.value)
            elif self.op == Binop.NUM_EQ:
                env[self.dest] = SBool(lhs == rhs)
            elif self.op == Binop.NUM_LT:
                env[self.dest] = SBool(lhs < rhs)
            else:
                raise ValueError(f"Unexpected op {self.op}")

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        # Type-based transfer function
        if self.op in (Binop.ADD, Binop.SUB, Binop.MUL, Binop.DIV, Binop.MOD):
            types[self.dest] = scheme_types.SchemeNum
        elif self.op in (Binop.SYM_EQ, Binop.PTR_EQ,
                         Binop.NUM_EQ, Binop.NUM_LT):
            types[self.dest] = scheme_types.SchemeBool
        else:
            raise ValueError(f"Unexpected op {self.op}")

        lhs, rhs = values[self.lhs], values[self.rhs]
        values[self.dest] = None
        if lhs is None or rhs is None:
            # Cannot do any constant folding
            return

        if self.op == Binop.SYM_EQ:
            if not (isinstance(lhs, SSym) and isinstance(rhs, SSym)):
                print("Unexpected args to SYM_EQ {self} ({lhs}, {rhs})")
                return

            types[self.dest] = scheme_types.SchemeBool
            values[self.dest] = SBool(lhs == rhs)
        elif self.op == Binop.PTR_EQ:
            types[self.dest] = scheme_types.SchemeBool
            values[self.dest] = SBool(lhs.address() == rhs.address())
        else:
            if not (isinstance(lhs, SNum) and isinstance(rhs, SNum)):
                print("Unexpected args to arith {self} ({lhs}, {rhs})")
                return

            res: Value
            if self.op == Binop.ADD:
                res = SNum(lhs.value + rhs.value)
            elif self.op == Binop.SUB:
                res = SNum(lhs.value - rhs.value)
            elif self.op == Binop.MUL:
                res = SNum(lhs.value * rhs.value)
            elif self.op == Binop.DIV:
                if rhs.value == 0:
                    print("Unexpected div by zero {self} ({rhs})")
                    return
                res = SNum(lhs.value // rhs.value)
            elif self.op == Binop.MOD:
                if rhs.value == 0:
                    print("Unexpected mod by zero {self} ({rhs})")
                    return
                res = SNum(lhs.value % rhs.value)
            elif self.op == Binop.NUM_EQ:
                res = SBool(lhs == rhs)
            elif self.op == Binop.NUM_LT:
                res = SBool(lhs < rhs)
            else:
                raise ValueError(f"Unexpected op {self.op}")
            values[self.dest] = res
            types[self.dest] = res.scheme_type()

    def __str__(self) -> str:
        return f"{self.dest} = {self.op} {self.lhs} {self.rhs}"

    def freshen(self, prefix: str) -> None:
        self.dest = self.dest.freshen(prefix)
        self.lhs = self.lhs.freshen(prefix)
        self.rhs = self.rhs.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> BinopInst:
        return BinopInst(self.dest, self.op,
                         values.get_param(self.lhs),
                         values.get_param(self.rhs))

    def dests(self) -> List[Var]:
        return [self.dest]

    def params(self) -> List[Parameter]:
        return [self.lhs, self.rhs]

    def pure(self) -> bool:
        return self.op not in (Binop.DIV, Binop.MOD)


@dataclass
class TypeofInst(Inst):
    dest: Var
    value: Parameter

    def run(self, env: EvalEnv) -> None:
        env[self.dest] = env[self.value].type_name()

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        val = values[self.value]
        if val is not None:
            values[self.dest] = val.type_name()
        else:
            values[self.dest] = types[self.value].symbol()

    def __str__(self) -> str:
        return f"{self.dest} = typeof {self.value}"

    def freshen(self, prefix: str) -> None:
        self.dest = self.dest.freshen(prefix)
        self.value = self.value.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> TypeofInst:
        return TypeofInst(self.dest, values.get_param(self.value))

    def dests(self) -> List[Var]:
        return [self.dest]

    def params(self) -> List[Parameter]:
        return [self.value]

    def pure(self) -> bool:
        return True


@dataclass
class CopyInst(Inst):
    dest: Var
    value: Parameter

    def run(self, env: EvalEnv) -> None:
        env[self.dest] = env[self.value]

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        types[self.dest] = types[self.value]
        values[self.dest] = values[self.value]

    def __str__(self) -> str:
        return f"{self.dest} = {self.value}"

    def freshen(self, prefix: str) -> None:
        self.dest = self.dest.freshen(prefix)
        self.value = self.value.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> CopyInst:
        return CopyInst(self.dest, values.get_param(self.value))

    def dests(self) -> List[Var]:
        return [self.dest]

    def params(self) -> List[Parameter]:
        return [self.value]

    def pure(self) -> bool:
        return True


@dataclass
class LookupInst(Inst):
    dest: Var
    name: Parameter

    def run(self, env: EvalEnv) -> None:
        sym = env[self.name]
        assert isinstance(sym, SSym)
        value = env._global_env[sym]
        assert isinstance(value, Value)
        env[self.dest] = value

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        # We don't know anything about globals.
        # @TODO: Know something about globals.
        types[self.dest] = scheme_types.SchemeObject
        values[self.dest] = None

    def __str__(self) -> str:
        return f"{self.dest} = lookup {self.name}"

    def freshen(self, prefix: str) -> None:
        self.dest = self.dest.freshen(prefix)
        self.name = self.name.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> LookupInst:
        return LookupInst(self.dest, values.get_param(self.name))

    def dests(self) -> List[Var]:
        return [self.dest]

    def params(self) -> List[Parameter]:
        return [self.name]

    def pure(self) -> bool:
        return True


@dataclass
class AllocInst(Inst):
    dest: Var
    size: Parameter

    def run(self, env: EvalEnv) -> None:
        size = env[self.size]
        assert isinstance(size, SNum)
        env[self.dest] = SVect([sexp.Nil] * size.value)

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        size = values[self.size]
        types[self.dest] = scheme_types.SchemeVectType(None)
        if size is None:
            values[self.dest] = None
        elif isinstance(size, SNum):
            types[self.dest] = scheme_types.SchemeVectType(size.value)
            values[self.dest] = SVect([sexp.Nil] * size.value)
        else:
            values[self.dest] = None
            print(f"Unexpected abstract AllocInst param {self} ({size})")

    def __str__(self) -> str:
        return f"{self.dest} = alloc {self.size}"

    def freshen(self, prefix: str) -> None:
        self.dest = self.dest.freshen(prefix)
        self.size = self.size.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> AllocInst:
        return AllocInst(self.dest, values.get_param(self.size))

    def dests(self) -> List[Var]:
        return [self.dest]

    def params(self) -> List[Parameter]:
        return [self.size]

    def pure(self) -> bool:
        return True


@dataclass
class LoadInst(Inst):
    dest: Var
    addr: Parameter
    offset: Parameter

    def run(self, env: EvalEnv) -> None:
        vect = env[self.addr]
        index = env[self.offset]
        assert isinstance(vect, SVect) and isinstance(index, SNum)
        assert index.value < len(vect.items)
        value = vect.items[index.value]
        assert isinstance(value, Value)
        env[self.dest] = value

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        offset = values[self.offset]
        vect = values[self.addr]
        if offset is None or vect is None:
            types[self.dest] = scheme_types.SchemeObject
            values[self.dest] = None
            return
        if isinstance(offset, SNum) and isinstance(vect, SVect):
            result = vect.items[offset.value]
            assert isinstance(result, Value)
            types[self.dest] = result.scheme_type()
            values[self.dest] = result
        else:
            types[self.dest] = scheme_types.SchemeObject
            values[self.dest] = None
            print(f"Warning! Unexpected abstract LoadInst {self}.")

    def __str__(self) -> str:
        return f"{self.dest} = load [{self.addr} + {self.offset}]"

    def freshen(self, prefix: str) -> None:
        self.dest = self.dest.freshen(prefix)
        self.addr = self.addr.freshen(prefix)
        self.offset = self.offset.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> LoadInst:
        return LoadInst(self.dest,
                        values.get_param(self.addr),
                        values.get_param(self.offset))

    def dests(self) -> List[Var]:
        return [self.dest]

    def params(self) -> List[Parameter]:
        return [self.addr, self.offset]

    def pure(self) -> bool:
        return True


@dataclass
class StoreInst(Inst):
    addr: Var
    offset: Parameter
    value: Parameter

    def run(self, env: EvalEnv) -> None:
        vect = env[self.addr]
        index = env[self.offset]
        assert isinstance(vect, SVect) and isinstance(index, SNum)
        assert index.value < len(vect.items)
        vect.items[index.value] = env[self.value]

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        vect = values[self.addr]
        offset = values[self.offset]
        value = values[self.value]
        if vect is None:
            # Disaster! We have to invalidate all vectors in values
            for k, value in values.values.items():
                if isinstance(value, SVect):
                    values[k] = None
        elif offset is None or value is None:
            values[self.addr] = None
        elif isinstance(offset, SNum) and isinstance(vect, SVect):
            vect.items[offset.value] = value

    def __str__(self) -> str:
        return f"store [{self.addr} + {self.offset}] = {self.value}"

    def freshen(self, prefix: str) -> None:
        self.addr = self.addr.freshen(prefix)
        self.offset = self.offset.freshen(prefix)
        self.value = self.value.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> StoreInst:
        return StoreInst(self.addr,
                         values.get_param(self.offset),
                         values.get_param(self.value))

    def dests(self) -> List[Var]:
        return []

    def params(self) -> List[Parameter]:
        return [self.addr, self.offset, self.value]

    def pure(self) -> bool:
        return False


@dataclass
class LengthInst(Inst):
    dest: Var
    addr: Parameter

    def run(self, env: EvalEnv) -> None:
        vect = env[self.addr]
        assert isinstance(vect, SVect), vect
        env[self.dest] = SNum(len(vect.items))

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        ty = types[self.dest]
        types[self.dest] = scheme_types.SchemeNum
        vect = values[self.addr]
        if vect is None:
            values[self.dest] = None
            if isinstance(ty, scheme_types.SchemeVectType) and ty.length:
                values[self.dest] = SNum(ty.length)
        elif isinstance(vect, SVect):
            values[self.dest] = SNum(len(vect.items))
        else:
            print(f"Unexpected length param {self} ({vect})")

    def __str__(self) -> str:
        return f"{self.dest} = length {self.addr}"

    def freshen(self, prefix: str) -> None:
        self.dest = self.dest.freshen(prefix)
        self.addr = self.addr.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> LengthInst:
        return copy.copy(self)

    def dests(self) -> List[Var]:
        return [self.dest]

    def params(self) -> List[Parameter]:
        return [self.addr]

    def pure(self) -> bool:
        return True


@dataclass
class ArityInst(Inst):
    dest: Var
    func: Parameter

    def run(self, env: EvalEnv) -> None:
        func = env[self.func]
        assert isinstance(func, sexp.SFunction), func
        env[self.dest] = SNum(len(func.params))

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        ty = types[self.func]
        val = values[self.func]
        types[self.dest] = scheme_types.SchemeNum
        if isinstance(ty, SchemeFunctionType) and ty.arity is not None:
            values[self.dest] = SNum(ty.arity)
        elif isinstance(val, sexp.SFunction):
            values[self.dest] = SNum(len(val.params))
        else:
            values[self.dest] = None

    def __str__(self) -> str:
        return f"{self.dest} = arity {self.func}"

    def freshen(self, prefix: str) -> None:
        self.dest = self.dest.freshen(prefix)
        self.func = self.func.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> ArityInst:
        return ArityInst(self.dest, values.get_param(self.func))

    def dests(self) -> List[Var]:
        return [self.dest]

    def params(self) -> List[Parameter]:
        return [self.func]

    def pure(self) -> bool:
        return True


@dataclass
class CallInst(Inst):
    dest: Var
    func: Parameter
    args: List[Parameter]
    specialization: Optional[TypeTuple] = None

    def run(self, env: EvalEnv) -> None:
        for _ in self.run_call(env):
            pass

    def run_call(self, env: EvalEnv) -> Generator[EvalEnv, None, None]:
        func = env[self.func]
        assert isinstance(func, sexp.SFunction)
        if func.code is None:
            raise NotImplementedError("JIT compiling functions!")
        func_code = func.code
        func_env = env.copy()
        assert len(func_code.params) == len(self.args)
        func_env._local_env = {
            name: env[arg] for name, arg in zip(func_code.params, self.args)
        }
        if self.specialization:
            assert self.specialization in func.specializations
            specialized = func.specializations[self.specialization]
            env[self.dest] = yield from specialized.run(func_env)
        else:
            env.stats.specialization_dispatch[id(self)] += 1
            type_tuple = tuple(env[arg].scheme_type() for arg in self.args)
            specialized = func.get_specialized(type_tuple)
            env[self.dest] = yield from specialized.run(func_env)

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        types[self.dest] = scheme_types.SchemeObject
        values[self.dest] = None

    def __str__(self) -> str:
        args = ', '.join(str(arg) for arg in self.args)
        text = f"{self.dest} = call {self.func} ({args})"
        if self.specialization:
            types = ', '.join(str(ty) for ty in self.specialization)
            text += f" ({types})"
        return text

    def freshen(self, prefix: str) -> None:
        self.dest = self.dest.freshen(prefix)
        self.func = self.func.freshen(prefix)
        for i in range(len(self.args)):
            self.args[i] = self.args[i].freshen(prefix)

    def constant_fold(self, values: ValueMap) -> CallInst:
        # @TODO: Improve the specialization
        return CallInst(self.dest,
                        values.get_param(self.func, allow_func=True),
                        [values.get_param(arg) for arg in self.args],
                        self.specialization)

    def dests(self) -> List[Var]:
        return [self.dest]

    def params(self) -> List[Parameter]:
        return [self.func] + self.args

    def pure(self) -> bool:
        return False


@dataclass
class JmpInst(Inst):
    target: BasicBlock

    def __repr__(self) -> str:
        return f"JmpInst(target={self.target.name})"

    def run(self, env: EvalEnv) -> BB:
        return self.target

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        pass

    def successors(self) -> Iterable[BasicBlock]:
        return [self.target]

    def __str__(self) -> str:
        return f"jmp {self.target.name}"

    def freshen(self, prefix: str) -> None:
        pass

    def constant_fold(self, values: ValueMap) -> JmpInst:
        return copy.copy(self)

    def dests(self) -> List[Var]:
        return []

    def params(self) -> List[Parameter]:
        return []

    def pure(self) -> bool:
        return False


@dataclass
class BrInst(Inst):
    cond: Parameter
    target: BasicBlock

    def __repr__(self) -> str:
        return (f"BrInst(cond={self.cond}, target={self.target.name})")

    def run(self, env: EvalEnv) -> Optional[BB]:
        res = env[self.cond]
        assert isinstance(res, sexp.SBool)
        if res.value:
            return self.target
        return None

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        pass

    def successors(self) -> Iterable[BasicBlock]:
        return [self.target]

    def __str__(self) -> str:
        return f"br {self.cond} {self.target.name}"

    def freshen(self, prefix: str) -> None:
        self.cond = self.cond.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> BrInst:
        return BrInst(values.get_param(self.cond), self.target)

    def dests(self) -> List[Var]:
        return []

    def params(self) -> List[Parameter]:
        return [self.cond]

    def pure(self) -> bool:
        return False


@dataclass
class BrnInst(Inst):
    cond: Parameter
    target: BasicBlock

    def __repr__(self) -> str:
        return (f"BrnInst(cond={self.cond}, target={self.target.name})")

    def run(self, env: EvalEnv) -> Optional[BB]:
        res = env[self.cond]
        assert isinstance(res, sexp.SBool)
        if not res.value:
            return self.target
        return None

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        pass

    def successors(self) -> Iterable[BasicBlock]:
        return [self.target]

    def __str__(self) -> str:
        return f"brn {self.cond} {self.target.name}"

    def freshen(self, prefix: str) -> None:
        self.cond = self.cond.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> BrnInst:
        return BrnInst(values.get_param(self.cond), self.target)

    def dests(self) -> List[Var]:
        return []

    def params(self) -> List[Parameter]:
        return [self.cond]

    def pure(self) -> bool:
        return False


@dataclass
class ReturnInst(Inst):
    ret: Parameter

    def run(self, env: EvalEnv) -> Optional[BB]:
        return ReturnBlock(f"return {self.ret}", self.ret)

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        pass

    def __str__(self) -> str:
        return f"return {self.ret}"

    def freshen(self, prefix: str) -> None:
        self.ret = self.ret.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> ReturnInst:
        return ReturnInst(values.get_param(self.ret))

    def dests(self) -> List[Var]:
        return []

    def params(self) -> List[Parameter]:
        return [self.ret]

    def pure(self) -> bool:
        return False


@dataclass
class TrapInst(Inst):
    message: str

    def run(self, env: EvalEnv) -> None:
        raise Trap(self.message)

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        pass

    def __str__(self) -> str:
        return f"trap {self.message!r}"

    def freshen(self, prefix: str) -> None:
        pass

    def constant_fold(self, values: ValueMap) -> TrapInst:
        return copy.copy(self)

    def dests(self) -> List[Var]:
        return []

    def params(self) -> List[Parameter]:
        return []

    def pure(self) -> bool:
        return False


@dataclass
class TraceInst(Inst):
    value: Parameter

    def run(self, env: EvalEnv) -> None:
        print(env[self.value])

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        pass

    def __str__(self) -> str:
        return f"trace {self.value}"

    def freshen(self, prefix: str) -> None:
        self.value = self.value.freshen(prefix)

    def constant_fold(self, values: ValueMap) -> TraceInst:
        return TraceInst(values.get_param(self.value))

    def dests(self) -> List[Var]:
        return []

    def params(self) -> List[Parameter]:
        return [self.value]

    def pure(self) -> bool:
        return False


@dataclass
class BreakpointInst(Inst):
    def run(self, env: EvalEnv) -> None:
        breakpoint()

    def run_abstract(self, types: TypeMap, values: ValueMap) -> None:
        pass

    def __str__(self) -> str:
        return f"breakpoint"

    def freshen(self, prefix: str) -> None:
        pass

    def constant_fold(self, values: ValueMap) -> BreakpointInst:
        return copy.copy(self)

    def dests(self) -> List[Var]:
        return []

    def params(self) -> List[Parameter]:
        return []

    def pure(self) -> bool:
        return False


@dataclass
class BasicBlock(BB):
    name: str
    instructions: List[Inst] = field(default_factory=list)
    split_counter: int = 0

    def __str__(self) -> str:
        return f"{self.name}:\n" + "\n".join(
            f"  {i}" for i in self.instructions
        )

    def format_stats(self, stats: Stats) -> str:
        return f"{stats.block_count[id(self)]:>8} {self.name}:\n" + "\n".join(
            f"{stats.inst_count[id(i)]:>8} "
            f"  {str(i):<40} "
            f"""{f'(taken {stats.taken_count[id(i)]})'
                  if stats.taken_count[id(i)] else ''}"""
            for i in self.instructions
        )

    def add_inst(self, inst: Inst) -> None:
        self.instructions.append(inst)

    def run(self, env: EvalEnv) -> Generator[EvalEnv, None, BB]:
        env.stats.block_count[id(self)] += 1
        for inst in self.instructions:
            env.stats.inst_type_count[type(inst)] += 1
            env.stats.inst_count[id(inst)] += 1
            if isinstance(inst, CallInst):
                yield from inst.run_call(env)
            else:
                next_bb = inst.run(env)
            yield env.copy()
            if next_bb is not None:
                env.stats.taken_count[id(inst)] += 1
                break
        assert next_bb
        return next_bb

    def successors(self) -> Iterator[BasicBlock]:
        for inst in self.instructions:
            yield from inst.successors()

    def split_after(self, idx: int) -> BasicBlock:
        new_block = BasicBlock(f"{self.name}.split{self.split_counter}",
                               self.instructions[idx+1:])
        self.split_counter += 1
        self.instructions = self.instructions[:idx+1]
        self.add_inst(JmpInst(new_block))
        return new_block


@dataclass
class ReturnBlock(BB):
    name: str
    ret: Parameter


@dataclass
class Function:
    params: List[Var]
    start: BasicBlock

    def run(self, env: EvalEnv) -> Generator[EvalEnv, None, Value]:
        assert all(p in env for p in self.params)
        block: BB = self.start
        env.stats.function_count[id(self)] += 1
        while True:
            if isinstance(block, BasicBlock):
                block = yield from block.run(env)
            elif isinstance(block, ReturnBlock):
                return env[block.ret]
            else:
                raise NotImplementedError(f"Unexpected BB type: {type(block)}")

    def blocks(self) -> Iterator[BasicBlock]:
        """
        Iterate over the basic blocks in some order.
        """
        visited: Set[int] = set()
        blocks: Queue[BasicBlock] = Queue()
        blocks.put(self.start)
        while not blocks.empty():
            block = blocks.get()
            yield block
            visited.add(id(block))
            for b in block.successors():
                if id(b) not in visited:
                    visited.add(id(b))
                    blocks.put(b)

    def __str__(self) -> str:
        return (f"function (?{''.join(' ' + x.name for x in self.params)})"
                f" entry={self.start.name}\n"
                + '\n\n'.join(str(b) for b in self.blocks()))

    def format_stats(self, name: SSym, stats: Stats) -> str:
        params = ''.join(' ' + x.name for x in self.params)
        return (f"function ({name}{params})"
                f" entry={self.start.name}\n"
                + '\n\n'.join(b.format_stats(stats) for b in self.blocks()))


T = TypeVar('T')


class ResultGenerator(Generic[T]):
    """A class to get the result from running a generator."""
    gen: Generator[Any, Any, T]
    value: Optional[T]

    def __init__(self, gen: Generator[Any, Any, T]):
        self.gen = gen
        self.value = None

    def __iter__(self) -> Any:
        self.value = yield from self.gen

    def run(self) -> None:
        for _ in self:
            pass
