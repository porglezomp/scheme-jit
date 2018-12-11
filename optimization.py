import copy
from dataclasses import dataclass, field
from queue import Queue
from typing import DefaultDict, Dict, Iterator, List, Optional, Set, Tuple

import bytecode
from bytecode import (BasicBlock, BoolLit, BrInst, BrnInst, CallInst, CopyInst,
                      EvalEnv, FuncLit, Function, JmpInst, LookupInst,
                      ReturnInst, SymLit, TypeMap, TypeTuple, ValueMap, Var)
from scheme_types import SchemeObjectType
from sexp import SBool, SFunction, SSym, Value

Id = int
Edge = Tuple[BasicBlock, int, BasicBlock]


@dataclass
class FunctionOptimizer:
    func: Function
    prefix_counter: int = 0
    specialization: Optional[TypeTuple] = None
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

        if block is self.func.start and self.specialization is not None:
            assert len(self.func.params) == len(self.specialization)
            types = TypeMap(dict(zip(self.func.params, self.specialization)))
            pred_maps.append((types, ValueMap()))

        # Join all of those maps
        if pred_maps:
            types, values = copy.deepcopy(pred_maps[0])
            for ty, val in pred_maps:
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

    def apply_constant_info(self) -> None:
        if not self.info:
            self.compute_dataflow()
        assert self.info
        for block in self.func.blocks():
            info_map = self.info[id(block)]
            for i, inst in enumerate(block.instructions):
                _, values = info_map[i]
                block.instructions[i] = inst.constant_fold(values)

    def find_lookups(self) -> Iterator[Tuple[BasicBlock, int, LookupInst]]:
        for block in self.func.blocks():
            for i, inst in enumerate(block.instructions):
                if isinstance(inst, LookupInst):
                    yield block, i, inst

    def find_calls(self) -> Iterator[Tuple[BasicBlock, int, CallInst]]:
        for block in self.func.blocks():
            for i, inst in enumerate(block.instructions):
                if isinstance(inst, CallInst):
                    yield block, i, inst

    def remove_dead_code(self) -> None:
        for block in self.func.blocks():
            for i in reversed(range(len(block.instructions))):
                inst = block.instructions[i]
                if isinstance(inst, BrInst) or isinstance(inst, BrnInst):
                    will_jump = isinstance(inst, BrInst)
                    if inst.cond == BoolLit(SBool(will_jump)):
                        block.instructions[i] = JmpInst(inst.target)
                    elif inst.cond == BoolLit(SBool(not will_jump)):
                        block.instructions.pop(i)
                elif inst.pure():
                    if not any(self.is_used(x, block) for x in inst.dests()):
                        # Its result is read nowhere, it can be deleted.
                        block.instructions.pop(i)

    def is_used(self, var: Var, block: Optional[BasicBlock] = None) -> bool:
        # Check `block` first, things are more likely to be used locally.
        if block is not None:
            for inst in block.instructions:
                if var in inst.params():
                    return True
        for block in self.func.blocks():
            for inst in block.instructions:
                if var in inst.params():
                    return True
        return False

    def mark_vars(self, func: Function) -> Function:
        func = copy.deepcopy(func)
        prefix = f"inl{self.prefix_counter}"
        self.prefix_counter += 1
        func.params = [p.freshen(prefix) for p in func.params]
        for block in func.blocks():
            block.name = f"{prefix}@{block.name}"
            for inst in block.instructions:
                inst.freshen(prefix)
        return func

    def should_inline(self, name: SSym) -> bool:
        SHOULD_INLINE = ('assert', 'number?')
        return name.name.startswith('inst/') or name.name in SHOULD_INLINE

    def seed_inlining(self, env: EvalEnv) -> None:
        for b, i, l in self.find_lookups():
            if isinstance(l.name, SymLit) and self.should_inline(l.name.value):
                func = env._global_env.get(l.name.value, None)
                if func is None:
                    print(f"failed to find {l.name} for inlining")
                    continue
                assert isinstance(func, SFunction)
                b.instructions[i] = bytecode.CopyInst(l.dest, FuncLit(func))

    def inline(self, env: EvalEnv) -> None:
        for block in self.func.blocks():
            for i in reversed(range(len(block.instructions))):
                inst = block.instructions[i]
                if not isinstance(inst, CallInst):
                    continue
                if not isinstance(inst.func, FuncLit):
                    continue
                next_block = block.split_after(i)
                func_code = inst.func.func.get_specialized(inst.specialization)
                func_code = self.mark_vars(func_code)
                for b in func_code.blocks():
                    for j, ret in enumerate(b.instructions):
                        if isinstance(ret, ReturnInst):
                            b.instructions[j] = CopyInst(inst.dest, ret.ret)
                            b.instructions.insert(j+1, JmpInst(next_block))
                block.instructions.pop()  # Remove the jmp
                block.instructions.pop()  # Remove the call
                for dst, src in zip(func_code.params, inst.args):
                    block.instructions.append(CopyInst(dst, src))
                block.instructions.append(JmpInst(func_code.start))
        self.preds = None
        self.succs = None
        self.info = None

    def merge_blocks(self) -> None:
        if self.preds is None:
            self.compute_preds()
        assert self.preds
        pass

    def legalize(self) -> None:
        for block in self.func.blocks():
            for i, inst in enumerate(block.instructions):
                if isinstance(inst, CopyInst):
                    if isinstance(inst.value, FuncLit):
                        block.instructions[i] = LookupInst(
                            inst.dest, SymLit(inst.value.func.name))
                assert not any(isinstance(p, FuncLit)
                               for p in block.instructions[i].params())
