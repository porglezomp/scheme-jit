import copy
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional, Tuple

import bytecode
from bytecode import BasicBlock, Function, TypeMap, ValueMap, Var
from scheme_types import SchemeObjectType
from sexp import Value

Id = int
Edge = Tuple[BasicBlock, int, BasicBlock]


@dataclass
class FunctionOptimizer:
    func: Function
    prefix_counter: int = 0
    succs: Optional[DefaultDict[Id, List[Edge]]] = None
    preds: Optional[DefaultDict[Id, List[Edge]]] = None

    def compute_preds(self) -> None:
        if not self.succs:
            self.compute_succs()
        assert self.succs
        self.preds = DefaultDict(list)
        for block in self.succs.keys():
            self.preds[block] = []
        for block, succs in self.succs.items():
            for (src, i, dst) in succs:
                self.preds[id(dst)].append((src, i, dst))

    def compute_succs(self) -> None:
        self.succs = DefaultDict(list)
        for block in self.func.blocks():
            self.succs[id(block)] = []
            for i, inst in enumerate(block.instructions):
                for succ in inst.successors():
                    self.succs[id(block)].append((block, i, succ))

    def block_transfer(
        self, block: BasicBlock, types: TypeMap, values: ValueMap,
    ) -> Dict[int, Tuple[TypeMap, ValueMap]]:
        jumps = {}
        for i, inst in enumerate(block.instructions):
            inst.run_abstract(types, values)
            if inst.successors():
                jumps[i] = (copy.deepcopy(types), copy.deepcopy(values))
        return jumps

    def mark_vars(self, func: Function) -> Function:
        func = copy.deepcopy(func)
        prefix = f"inl{self.prefix_counter}"
        for block in func.blocks():
            block.name = f"{prefix}@{block.name}"
            for inst in block.instructions:
                inst.freshen(prefix)
        return func

    def inline_at_block_tail(self, block: BasicBlock, func: Function) -> None:
        assert len(block.instructions) >= 2
        assert isinstance(block.instructions[-2], bytecode.CallInst)
        assert isinstance(block.instructions[-1], bytecode.JmpInst)
        raise NotImplementedError("inlining")
