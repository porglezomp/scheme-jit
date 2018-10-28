from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from scheme import SNum, SSym


class Value(ABC):
    pass


class Inst(ABC):
    pass


class TerminatorInst(Inst, ABC):
    pass


class BB(ABC):
    name: str


@dataclass(frozen=True)
class Var(Value):
    name: str


@dataclass(frozen=True)
class NumLit(Value):
    value: SNum


@dataclass(frozen=True)
class SymLit(Value):
    value: SSym


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
