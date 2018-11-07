from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (Counter, Dict, Generator, Iterable, Iterator, List,
                    Optional, Set)

import scheme
from environment import Environment
from errors import Trap
from scheme import SNum, SSym, SVect, Value


class Parameter(ABC):
    @abstractmethod
    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        ...


class Inst(ABC):
    @abstractmethod
    def run(self, env: EvalEnv) -> Optional[BB]:
        ...

    def successors(self) -> Iterable[BB]:
        return []


class BB(ABC):
    name: str

    def successors(self) -> Iterable[BB]:
        return []


@dataclass(frozen=True)
class Var(Parameter):
    name: str

    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        return env[self]


@dataclass(frozen=True)
class NumLit(Parameter):
    value: SNum

    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        return self.value


@dataclass(frozen=True)
class SymLit(Parameter):
    value: SSym

    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        return self.value


class EvalEnv:
    _local_env: Dict[Var, Value]
    _global_env: Environment
    stats: Counter[type]

    def __init__(self,
                 local_env: Optional[Dict[Var, Value]] = None,
                 global_env: Optional[Environment] = None):
        if local_env is None:
            self._local_env = {}
        else:
            self._local_env = local_env
        if global_env is None:
            self._global_env = Environment(None)
        else:
            self._global_env = global_env
        self.stats = Counter()

    def copy(self) -> EvalEnv:
        """Return a shallow copy of the environment."""
        env = EvalEnv(self._local_env.copy(), self._global_env)
        env.stats = self.stats.copy()
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
        env.stats[type(self)] += 1
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
    value: Parameter

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        value = env[self.value]
        if isinstance(value, SNum):
            env[self.dest] = SSym('number')
        elif isinstance(value, SSym):
            env[self.dest] = SSym('symbol')
        elif isinstance(value, SVect):
            env[self.dest] = SSym('vector')
        else:
            raise ValueError(f"Parameter {value} wasn't an expected type.")


@dataclass
class CopyInst(Inst):
    dest: Var
    value: Parameter

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        env[self.dest] = env[self.value]


@dataclass
class LookupInst(Inst):
    dest: Var
    name: Parameter

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        sym = env[self.name]
        assert isinstance(sym, SSym)
        value = env._global_env[sym]
        assert isinstance(value, Value)
        env[self.dest] = value


@dataclass
class AllocInst(Inst):
    dest: Var
    size: Parameter

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        size = env[self.size]
        assert isinstance(size, SNum)
        env[self.dest] = SVect([scheme.Nil] * size.value)


@dataclass
class LoadInst(Inst):
    dest: Var
    addr: Parameter
    offset: Parameter

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        vect = env[self.addr]
        index = env[self.offset]
        assert isinstance(vect, SVect) and isinstance(index, SNum)
        assert index.value < len(vect.items)
        value = vect.items[index.value]
        assert isinstance(value, Value)
        env[self.dest] = value


@dataclass
class StoreInst(Inst):
    addr: Parameter
    offset: Parameter
    value: Parameter

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        vect = env[self.addr]
        index = env[self.offset]
        assert isinstance(vect, SVect) and isinstance(index, SNum)
        assert index.value < len(vect.items)
        vect.items[index.value] = env[self.value]


@dataclass
class LengthInst(Inst):
    dest: Var
    addr: Parameter

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        vect = env[self.addr]
        assert isinstance(vect, SVect)
        env[self.dest] = SNum(len(vect.items))


@dataclass
class CallInst(Inst):
    dest: Var
    name: Parameter
    args: List[Parameter]

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
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
class JmpInst(Inst):
    target: BB

    def __repr__(self) -> str:
        return f"JmpInst(target={self.target.name})"

    def run(self, env: EvalEnv) -> BB:
        env.stats[type(self)] += 1
        return self.target

    def successors(self) -> Iterable[BB]:
        return [self.target]


@dataclass
class BrInst(Inst):
    cond: Parameter
    target: BB

    def __repr__(self) -> str:
        return (f"BrInst(cond={self.cond}, target={self.target.name})")

    def run(self, env: EvalEnv) -> Optional[BB]:
        env.stats[type(self)] += 1
        res = env[self.cond]
        assert res in (SSym('true'), SSym('false'))
        if res == SSym('true'):
            return self.target
        return None

    def successors(self) -> Iterable[BB]:
        return [self.target]


@dataclass
class BrnInst(Inst):
    cond: Parameter
    target: BB

    def __repr__(self) -> str:
        return (f"BrnInst(cond={self.cond}, target={self.target.name})")

    def run(self, env: EvalEnv) -> Optional[BB]:
        env.stats[type(self)] += 1
        res = env[self.cond]
        assert res in (SSym('true'), SSym('false'))
        if res == SSym('false'):
            return self.target
        return None

    def successors(self) -> Iterable[BB]:
        return [self.target]


@dataclass
class ReturnInst(Inst):
    ret: Parameter

    def run(self, env: EvalEnv) -> Optional[BB]:
        return ReturnBlock(f"return {self.ret}", self.ret)


@dataclass
class TrapInst(Inst):
    message: str

    def run(self, env: EvalEnv) -> None:
        raise Trap(self.message)


@dataclass
class BasicBlock(BB):
    name: str
    instructions: List[Inst] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.name}:\n" + "\n".join(
            "  " + str(i) for i in self.instructions
        )

    def add_inst(self, inst: Inst) -> None:
        self.instructions.append(inst)

    def run(self, env: EvalEnv) -> Generator[EvalEnv, None, BB]:
        env.stats[type(self)] += 1
        for inst in self.instructions:
            if isinstance(inst, CallInst):
                yield from inst.run_call(env)
            next_bb = inst.run(env)
            yield env.copy()
            if next_bb is not None:
                break
        assert next_bb
        return next_bb

    def successors(self) -> Iterator[BB]:
        for inst in self.instructions:
            yield from inst.successors()


@dataclass
class ReturnBlock(BB):
    name: str
    ret: Parameter


@dataclass
class Function:
    params: List[Var]
    start: BB

    def run(self, env: EvalEnv) -> Generator[EvalEnv, None, Value]:
        assert all(p in env for p in self.params)
        block = self.start
        while True:
            if isinstance(block, BasicBlock):
                block = yield from block.run(env)
            elif isinstance(block, ReturnBlock):
                return env[block.ret]
            else:
                raise NotImplementedError(f"Unexpected BB type: {type(block)}")

    def blocks(self) -> Iterator[BB]:
        """
        Iterate over the basic blocks in some order.

        Ideally this would be the preorder traversal of the dom-tree.
        """
        visited: Set[int] = set()
        blocks = [self.start]
        while blocks:
            block = blocks.pop()
            yield block
            blocks.extend(b for b in block.successors()
                          if id(b) not in visited)
            visited |= set(map(id, blocks))

    def __str__(self) -> str:
        return (f"def ({', '.join(x.name for x in self.params)})\n"
                + '\n\n'.join(str(b) for b in self.blocks()))
