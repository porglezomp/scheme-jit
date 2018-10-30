from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from environment import Environment
from scheme import SExp, SNum, SSym


class Value(ABC):
    @abstractmethod
    def lookup_self(self, env: Dict[Var, SExp]) -> SExp:
        ...


class Inst(ABC):
    pass


class TerminatorInst(Inst, ABC):
    pass


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
    op: Binop
    dest: Var
    lhs: Value
    rhs: Value


@dataclass
class TypeofInst(Inst):
    dest: Var
    value: Value


@dataclass
class CopyInst(Inst):
    dest: Var
    value: Value


@dataclass
class LookupInst(Inst):
    dest: Var
    name: Value


@dataclass
class AllocInst(Inst):
    dest: Var
    size: Value


@dataclass
class LoadInst(Inst):
    dest: Var
    addr: Value


@dataclass
class StoreInst(Inst):
    addr: Value
    value: Value


@dataclass
class CallInst(Inst):
    dest: Var
    name: Value
    args: List[Value]


@dataclass
class Jmp(TerminatorInst):
    target: BB

    def __repr__(self) -> str:
        return f"Jmp(target={self.target.name})"


@dataclass
class Br(TerminatorInst):
    cond: Value
    then_target: BB
    else_target: BB

    def __repr__(self) -> str:
        return (f"Br(cond={self.cond}, "
                "then_target={self.then_target.name}, "
                "else_target={self.else_target.name})")


@dataclass
class BasicBlock(BB):
    name: str
    terminator: Optional[TerminatorInst] = None
    instructions: List[Inst] = field(default_factory=list)

    def add_inst(self, inst: Inst) -> None:
        self.instructions.append(inst)


@dataclass
class ReturnBlock(BB):
    name: str
    ret: Var


@dataclass
class Function:
    params: List[Var]
    start: BasicBlock
    finish: ReturnBlock
