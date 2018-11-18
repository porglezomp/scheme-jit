from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (Any, Counter, Dict, Generator, Generic, Iterable, Iterator,
                    List, Optional, Set, TypeVar)

import scheme
from errors import Trap
from scheme import SBool, SExp, SNum, SSym, SVect, Value


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

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class NumLit(Parameter):
    value: SNum

    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        return self.value

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class SymLit(Parameter):
    value: SSym

    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        return self.value

    def __str__(self) -> str:
        return f"'{self.value}"


@dataclass(frozen=True)
class BoolLit(Parameter):
    value: SBool

    def lookup_self(self, env: Dict[Var, Value]) -> Value:
        return self.value

    def __str__(self) -> str:
        return f"'{self.value}"


class EvalEnv:
    _local_env: Dict[Var, Value]
    _global_env: Dict[SSym, Value]
    stats: Counter[type]

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
            env[self.dest] = scheme.SBool(lhs == rhs)
        elif self.op == Binop.PTR_EQ:
            env[self.dest] = scheme.SBool(lhs.address() == rhs.address())
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
                env[self.dest] = scheme.SBool(lhs == rhs)
            elif self.op == Binop.NUM_LT:
                env[self.dest] = scheme.SBool(lhs < rhs)
            else:
                raise ValueError(f"Unexpected op {self.op}")

    def __str__(self) -> str:
        return f"{self.dest} = {self.op} {self.lhs} {self.rhs}"


@dataclass
class TypeofInst(Inst):
    dest: Var
    value: Parameter

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        env[self.dest] = env[self.value].type_name()

    def __str__(self) -> str:
        return f"{self.dest} = typeof {self.value}"


@dataclass
class CopyInst(Inst):
    dest: Var
    value: Parameter

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        env[self.dest] = env[self.value]

    def __str__(self) -> str:
        return f"{self.dest} = {self.value}"


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

    def __str__(self) -> str:
        return f"{self.dest} = lookup {self.name}"


@dataclass
class AllocInst(Inst):
    dest: Var
    size: Parameter

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        size = env[self.size]
        assert isinstance(size, SNum)
        env[self.dest] = SVect([scheme.Nil] * size.value)

    def __str__(self) -> str:
        return f"{self.dest} = alloc {self.size}"


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

    def __str__(self) -> str:
        return f"{self.dest} = load [{self.addr} + {self.offset}]"


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

    def __str__(self) -> str:
        return f"store [{self.addr} + {self.offset}] = {self.value}"


@dataclass
class LengthInst(Inst):
    dest: Var
    addr: Parameter

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        vect = env[self.addr]
        assert isinstance(vect, SVect), vect
        env[self.dest] = SNum(len(vect.items))

    def __str__(self) -> str:
        return f"{self.dest} = length {self.addr}"


@dataclass
class CallInst(Inst):
    dest: Var
    func: Parameter
    args: List[Parameter]

    def run(self, env: EvalEnv) -> None:
        env.stats[type(self)] += 1
        for _ in self.run_call(env):
            pass

    def run_call(self, env: EvalEnv) -> Generator[EvalEnv, None, None]:
        func = env[self.func]
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

    def __str__(self) -> str:
        args = ', '.join(str(arg) for arg in self.args)
        return f"{self.dest} = call {self.func} ({args})"


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

    def __str__(self) -> str:
        return f"jmp {self.target.name}"


@dataclass
class BrInst(Inst):
    cond: Parameter
    target: BB

    def __repr__(self) -> str:
        return (f"BrInst(cond={self.cond}, target={self.target.name})")

    def run(self, env: EvalEnv) -> Optional[BB]:
        env.stats[type(self)] += 1
        res = env[self.cond]
        assert isinstance(res, scheme.SBool)
        if res.value:
            return self.target
        return None

    def successors(self) -> Iterable[BB]:
        return [self.target]

    def __str__(self) -> str:
        return f"br {self.cond} {self.target.name}"


@dataclass
class BrnInst(Inst):
    cond: Parameter
    target: BB

    def __repr__(self) -> str:
        return (f"BrnInst(cond={self.cond}, target={self.target.name})")

    def run(self, env: EvalEnv) -> Optional[BB]:
        env.stats[type(self)] += 1
        res = env[self.cond]
        assert isinstance(res, scheme.SBool)
        if not res.value:
            return self.target
        return None

    def successors(self) -> Iterable[BB]:
        return [self.target]

    def __str__(self) -> str:
        return f"brn {self.cond} {self.target.name}"


@dataclass
class ReturnInst(Inst):
    ret: Parameter

    def run(self, env: EvalEnv) -> Optional[BB]:
        return ReturnBlock(f"return {self.ret}", self.ret)

    def __str__(self) -> str:
        return f"return {self.ret}"


@dataclass
class TrapInst(Inst):
    message: str

    def run(self, env: EvalEnv) -> None:
        raise Trap(self.message)

    def __str__(self) -> str:
        return f"trap {self.message!r}"


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
        return (f"function ({', '.join(x.name for x in self.params)})"
                f" entry={self.start.name}\n"
                + '\n\n'.join(str(b) for b in self.blocks()))


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
