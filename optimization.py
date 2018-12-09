import copy
from dataclasses import dataclass, field
from queue import Queue
from typing import DefaultDict, Dict, Iterator, List, Optional, Set, Tuple

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
    dominators: Optional[Dict[int, Set[BasicBlock]]] = None
    domtree: Optional[Dict[int, List[BasicBlock]]] = None
    info: Optional[Dict[int, List[Tuple[TypeMap, ValueMap]]]] = None

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

    def compute_dominators(self) -> None:
        raise NotImplementedError("dominators")

    def compute_domtree(self) -> None:
        if not self.dominators:
            self.compute_dominators()
        raise NotImplementedError("domtree")

    def dom_order_blocks(self) -> Iterator[BasicBlock]:
        """
        Iterate over the basic blocks.

        Visits them in a breadth-first traversal of the dominator tree.
        """
        if not self.domtree:
            self.compute_domtree()
        assert self.domtree
        blocks: Queue[BasicBlock] = Queue()
        blocks.put(self.func.start)
        while not blocks.empty():
            block = blocks.get()
            yield block
            for b in self.domtree[id(block)]:
                blocks.put(b)

    def block_transfer(
        self, block: BasicBlock, types: TypeMap, values: ValueMap,
    ) -> List[Tuple[TypeMap, ValueMap]]:
        abstract = []
        for i, inst in enumerate(block.instructions):
            abstract.append((copy.deepcopy(types), copy.deepcopy(values)))
            inst.run_abstract(types, values)
        abstract.append((copy.deepcopy(types), copy.deepcopy(values)))
        return abstract

    def block_input_maps(self, block: BasicBlock) -> Tuple[TypeMap, ValueMap]:
        assert self.preds is not None and self.info is not None
        # Find the maps for each incoming edge
        preds = self.preds[id(block)]
        pred_maps: List[Tuple[TypeMap, ValueMap]] = []
        for src, i, _ in preds:
            src_info = self.info.get(id(src), None)
            if src_info:
                pred_maps.append(src_info[i])
            else:
                pred_maps.append((TypeMap(), ValueMap()))

        # Join all of those maps
        if pred_maps:
            types, values = copy.deepcopy(pred_maps[0])
            for ty, val in pred_maps[1:]:
                types, values = types.join(ty), values.join(val)
        else:
            types, values = TypeMap(), ValueMap()
        return types, values

    def compute_dataflow(self) -> None:
        if not self.preds:
            self.compute_preds()
        assert self.preds
        if self.info is None:
            self.info = {}

        for block in self.func.blocks():
            types, values = self.block_input_maps(block)
            self.info[id(block)] = self.block_transfer(block, types, values)

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
