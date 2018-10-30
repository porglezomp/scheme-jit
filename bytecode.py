from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Generator, List, Optional

import scheme
from environment import Environment
from scheme import SExp, SNum, SSym, SVect


class Value(ABC):
    @abstractmethod
    def lookup_self(self, env: Dict[Var, SExp]) -> SExp:
        ...


class Inst(ABC):
    @abstractmethod
    def run(self, env: EvalEnv) -> None:
        ...


class TerminatorInst(ABC):
    @abstractmethod
    def run(self, env: EvalEnv) -> BB:
        ...


class BB(ABC):
    name: str


@dataclass(frozen=True)
class Var(Value):
    name: str

    def lookup_self(self, env: Dict[Var, SExp]) -> SExp:
        return env[self]


@dataclass(frozen=True)
class NumLit(Value):
    value: SNum

    def lookup_self(self, env: Dict[Var, SExp]) -> SExp:
        return self.value


@dataclass(frozen=True)
class SymLit(Value):
    value: SSym

    def lookup_self(self, env: Dict[Var, SExp]) -> SExp:
        return self.value


class EvalEnv:
    _local_env: Dict[Var, SExp]
    _global_env: Environment
    stats: Dict[type, int]

    def __init__(self,
                 local_env: Optional[Dict[Var, SExp]] = None,
                 global_env: Optional[Environment] = None):
        if local_env is None:
            self._local_env = {}
        else:
            self._local_env = local_env
        if global_env is None:
            self._global_env = Environment(None)
        else:
            self._global_env = global_env
        self.stats = {}

    def copy(self) -> EvalEnv:
        """Return a shallow copy of the environment."""
        env = EvalEnv(self._local_env.copy(), self._global_env)
        env.stats = self.stats.copy()
        return env

    def __getitem__(self, key: Value) -> SExp:
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

    def __setitem__(self, key: Var, value: SExp) -> None:
        self._local_env[key] = value

    def __contains__(self, key: Value) -> bool:
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
    lhs: Value
    rhs: Value

    def run(self, env: EvalEnv) -> None:
        lhs = env[self.lhs]
        rhs = env[self.rhs]
        if self.op == Binop.SYM_EQ:
            assert isinstance(lhs, SSym) and isinstance(rhs, SSym)
            env[self.dest] = scheme.make_bool(lhs == rhs)
        elif self.op == Binop.PTR_EQ:
            assert isinstance(lhs, SVect) and isinstance(rhs, SVect)
            env[self.dest] = scheme.make_bool(lhs is rhs)
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
                env[self.dest] = scheme.make_bool(lhs == rhs)
            elif self.op == Binop.NUM_LT:
                env[self.dest] = scheme.make_bool(lhs < rhs)
            else:
                raise ValueError(f"Unexpected op {self.op}")


@dataclass
class TypeofInst(Inst):
    dest: Var
    value: Value

    def run(self, env: EvalEnv) -> None:
        value = env[self.value]
        if isinstance(value, SNum):
            env[self.dest] = SSym('number')
        elif isinstance(value, SSym):
            env[self.dest] = SSym('symbol')
        elif isinstance(value, SVect):
            env[self.dest] = SSym('vector')
        else:
            raise ValueError(f"Value {value} wasn't an expected type.")


@dataclass
class CopyInst(Inst):
    dest: Var
    value: Value

    def run(self, env: EvalEnv) -> None:
        env[self.dest] = env[self.value]


@dataclass
class LookupInst(Inst):
    dest: Var
    name: Value

    def run(self, env: EvalEnv) -> None:
        sym = env[self.name]
        assert isinstance(sym, SSym)
        env[self.dest] = env._global_env[sym]


@dataclass
class AllocInst(Inst):
    dest: Var
    size: Value

    def run(self, env: EvalEnv) -> None:
        size = env[self.size]
        assert isinstance(size, SNum)
        env[self.dest] = SVect([scheme.Nil] * size.value)


@dataclass
class LoadInst(Inst):
    dest: Var
    addr: Value
    offset: Value

    def run(self, env: EvalEnv) -> None:
        vect = env[self.addr]
        index = env[self.offset]
        assert isinstance(vect, SVect) and isinstance(index, SNum)
        assert index.value < len(vect.items)
        env[self.dest] = vect.items[index.value]


@dataclass
class StoreInst(Inst):
    addr: Value
    offset: Value
    value: Value

    def run(self, env: EvalEnv) -> None:
        vect = env[self.addr]
        index = env[self.offset]
        assert isinstance(vect, SVect) and isinstance(index, SNum)
        assert index.value < len(vect.items)
        vect.items[index.value] = env[self.value]


@dataclass
class LengthInst(Inst):
    dest: Var
    addr: Value

    def run(self, env: EvalEnv) -> None:
        vect = env[self.addr]
        assert isinstance(vect, SVect)
        env[self.dest] = SNum(len(vect.items))


@dataclass
class CallInst(Inst):
    dest: Var
    name: Value
    args: List[Value]

    def run(self, env: EvalEnv) -> None:
        for _ in self.run_call(env):
            pass

    def run_call(self, env: EvalEnv) -> Generator[EvalEnv, None, None]:
        func = env[self.name]
        assert isinstance(func, scheme.SFunction)
        if func.code is None:
            raise NotImplementedError("JIT compiling functions!")
        func_code = func.code
        func_env = env.copy()
        assert len(func_code.params)
        func_env._local_env = {
            name: env[arg] for name, arg in zip(func_code.params, self.args)
        }
        env[self.dest] = yield from func_code.run(func_env)


@dataclass
class Jmp(TerminatorInst):
    target: BB

    def __repr__(self) -> str:
        return f"Jmp(target={self.target.name})"

    def run(self, env: EvalEnv) -> BB:
        return self.target


@dataclass
class Br(TerminatorInst):
    cond: Value
    then_target: BB
    else_target: BB

    def __repr__(self) -> str:
        return (f"Br(cond={self.cond}, "
                "then_target={self.then_target.name}, "
                "else_target={self.else_target.name})")

    def run(self, env: EvalEnv) -> BB:
        res = env[self.cond]
        if res == SSym('true'):
            return self.then_target
        elif res == SSym('false'):
            return self.else_target
        else:
            raise ValueError(f"Invalid boolean {res} in Br")


@dataclass
class BasicBlock(BB):
    name: str
    terminator: Optional[TerminatorInst] = None
    instructions: List[Inst] = field(default_factory=list)

    def add_inst(self, inst: Inst) -> None:
        self.instructions.append(inst)

    def run(self, env: EvalEnv) -> Generator[EvalEnv, None, BB]:
        for inst in self.instructions:
            if isinstance(inst, CallInst):
                yield from inst.run_call(env)
            inst.run(env)
            yield env.copy()
        assert self.terminator
        return self.terminator.run(env)


@dataclass
class ReturnBlock(BB):
    name: str
    ret: Value


@dataclass
class Function:
    params: List[Var]
    start: BasicBlock
    finish: ReturnBlock

    def run(self, env: EvalEnv) -> Generator[EvalEnv, None, SExp]:
        assert all(p in env for p in self.params)
        block: BB = self.start
        while isinstance(block, BasicBlock):
            block = yield from block.run(env)
        assert isinstance(block, ReturnBlock)
        return env[block.ret]
