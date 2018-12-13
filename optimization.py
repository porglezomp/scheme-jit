import copy
from dataclasses import dataclass, field
from queue import Queue
from typing import DefaultDict, Dict, Iterator, List, Optional, Set, Tuple

import bytecode
from bytecode import (BasicBlock, BoolLit, BrInst, BrnInst, CallInst, CopyInst,
                      EvalEnv, Function, Inst, JmpInst, LookupInst, Parameter,
                      ReturnInst, SymLit, TrapInst, TypeMap, TypeTuple,
                      ValueMap, Var)
from scheme_types import SchemeObject, SchemeObjectType
from sexp import SBool, SFunction, SSym, Value

Id = int
Edge = Tuple[BasicBlock, int, BasicBlock]


@dataclass
class FunctionOptimizer:
    func: Function
    prefix_counter: int = 0
    specialization: Optional[TypeTuple] = None
    inputs: Optional[Tuple[Optional[Value], ...]] = None
    succs: Optional[DefaultDict[Id, List[Edge]]] = None
    preds: Optional[DefaultDict[Id, List[Edge]]] = None
    dominators: Optional[Dict[int, Set[BasicBlock]]] = None
    domtree: Optional[Dict[int, List[BasicBlock]]] = None
    info: Optional[Dict[int, List[Tuple[TypeMap, ValueMap]]]] = None
    inlines: Dict[Id, Tuple[BasicBlock, Dict[int, Function]]] = (
        field(default_factory=dict)
    )
    result: Optional[Tuple[SchemeObjectType, Optional[Value]]] = None
    banned_from_inline: Set[SSym] = field(default_factory=set)

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
            self, env: EvalEnv, block: BasicBlock,
            types: TypeMap, values: ValueMap,
    ) -> List[Tuple[TypeMap, ValueMap]]:
        abstract = []
        for i, inst in enumerate(block.instructions):
            abstract.append((copy.copy(types), copy.copy(values)))
            inst.run_abstract(env, types, values)
            block.instructions[i] = inst.constant_fold(types, values)
            if isinstance(inst, CallInst):
                func = values[inst.func]
                if (isinstance(func, SFunction)
                    and self.should_inline(func, inst.specialization)
                ):  # noqa
                    code = func.get_specialized(inst.specialization)
                    code = copy.deepcopy(code)
                    opt = FunctionOptimizer(code)
                    opt.specialization = inst.specialization
                    opt.inputs = tuple(values[x] for x in inst.args)
                    opt.banned_from_inline = (
                        self.banned_from_inline | {func.name}
                    )
                    opt.optimize(env)
                    if opt.result is not None:
                        ty, val = opt.result
                        types[inst.dest] = ty
                        values[inst.dest] = val
                    if id(block) not in self.inlines:
                        self.inlines[id(block)] = (block, {})
                    self.inlines[id(block)][1][i] = code
            if isinstance(inst, ReturnInst):
                ret_ty = types[inst.ret]
                ret_val = values[inst.ret]
                if self.result is None:
                    self.result = (ret_ty, ret_val)
                else:
                    self.result = (
                        self.result[0].join(ret_ty),
                        self.result[1] if self.result[1] == ret_val else None
                    )
        abstract.append((copy.copy(types), copy.copy(values)))
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

        # Handle the specialization and known inputs of the function
        if block is self.func.start:
            types, values = TypeMap(), ValueMap()
            if self.specialization is not None:
                assert len(self.func.params) == len(self.specialization)
                types = TypeMap(dict(zip(self.func.params,
                                         self.specialization)))
            if self.inputs is not None:
                assert len(self.func.params) == len(self.inputs)
                values = ValueMap({
                    param: value
                    for param, value in zip(self.func.params, self.inputs)
                    if value is not None
                })
            pred_maps.append((types, values))

        # Join all of those maps
        if pred_maps:
            types, values = pred_maps[0]
            types, values = copy.copy(types), copy.copy(values)
            for ty, val in pred_maps:
                types, values = types.join(ty), values.join(val)
        else:
            types, values = TypeMap(), ValueMap()
        return types, values

    def dataflow(self, env: EvalEnv) -> None:
        if not self.preds:
            self.compute_preds()
        assert self.preds
        if self.info is None:
            self.info = {}

        for block in self.func.blocks():
            types, values = self.block_input_maps(block)
            self.info[id(block)] = self.block_transfer(
                env, block, types, values)
        self.apply_inlining(env)

    def remove_dead_code(self) -> None:
        for block in self.func.blocks():
            for i in reversed(range(len(block.instructions))):
                inst = block.instructions[i]
                if isinstance(inst, (BrInst, BrnInst)):
                    will_jump = isinstance(inst, BrInst)
                    if inst.cond == BoolLit(SBool(will_jump)):
                        block.instructions[i] = JmpInst(inst.target)
                    elif inst.cond == BoolLit(SBool(not will_jump)):
                        block.instructions.pop(i)
                elif isinstance(inst, (JmpInst, TrapInst)):
                    block.instructions = block.instructions[:i+1]
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

    def apply_inlining(self, env: EvalEnv) -> None:
        did_inline = False
        for block, inls in self.inlines.values():
            last_i = None
            for i, func in reversed(sorted(inls.items())):
                did_inline = True
                inst = block.instructions[i]
                assert isinstance(inst, CallInst)
                if last_i is not None:
                    assert i < last_i
                last_i = i
                func = self.mark_vars(func)
                next_block = block.split_after(i)
                # We precompute the list so it doesn't change as we modify
                # control flow.
                func_blocks = list(func.blocks())
                for b in func_blocks:
                    for j, ret in enumerate(b.instructions):
                        if isinstance(ret, ReturnInst):
                            b.instructions[j] = CopyInst(inst.dest, ret.ret)
                            b.instructions.insert(j+1, JmpInst(next_block))
                block.instructions.pop()  # Remove the jmp
                block.instructions.pop()  # Remove the call
                for dst, src in zip(func.params, inst.args):
                    block.instructions.append(CopyInst(dst, src))
                block.instructions.append(JmpInst(func.start))
        if did_inline:
            self.preds = None
            self.succs = None
            self.info = None

    def copy_propagate_block(self, block: BasicBlock) -> None:
        copies: Dict[Var, Parameter] = {}
        for i, inst in enumerate(block.instructions):
            if isinstance(inst, CopyInst):
                value = inst.value
                while value in copies:
                    assert isinstance(value, Var)
                    value = copies[value]
                copies[inst.dest] = value
            else:
                block.instructions[i] = inst.copy_prop(copies)

    def copy_propagate(self) -> None:
        for block in self.func.blocks():
            self.copy_propagate_block(block)

    def merge_blocks(self) -> None:
        if self.preds is None:
            self.compute_preds()
            assert self.preds

        def mergable(block: BasicBlock) -> bool:
            # @TODO: Branch switching for conditional+unconditional jumps.
            # It would be nice to be able to use the better one of the two
            # tails.
            assert self.preds
            last = block.instructions[-1]
            if isinstance(last, JmpInst):
                return len(self.preds[id(last.target)]) == 1
            elif isinstance(last, TrapInst) and len(block.instructions) > 1:
                prev = block.instructions[-2]
                if isinstance(prev, (BrInst, BrnInst)):
                    if not len(self.preds[id(prev.target)]) == 1:
                        return False
                    new_target = block.split_after(-2)
                    block.instructions[-1] = JmpInst(prev.target)
                    if isinstance(prev, BrInst):
                        block.instructions[-2] = BrnInst(prev.cond, new_target)
                    else:
                        block.instructions[-2] = BrInst(prev.cond, new_target)
                    return True
            return False

        for block in self.func.blocks():
            while mergable(block):
                last = block.instructions[-1]
                assert isinstance(last, JmpInst)
                block.instructions.pop()
                block.instructions.extend(last.target.instructions)
                del self.preds[id(last.target)]

        # @TODO: Repair dataflow info instead of invalidating it all?
        self.info = None

    def fmt_inst(self, inst: Inst, types: TypeMap, values: ValueMap) -> str:
        bindings = []
        for param in inst.params():
            if isinstance(param, Var):
                val = values[param]
                if val is not None:
                    bindings.append(f"{param} = {val}")
                elif types[param] != SchemeObject:
                    bindings.append(f"{param}: {types[param]}")
        if bindings:
            return f"{inst!s:<59} {{{', '.join(bindings)}}}"
        else:
            return str(inst)

    def print_func(self) -> None:
        if self.info is None:
            print(self.func)
            return

        print(f"function (?{''.join(' ' + x.name for x in self.func.params)})"
              f" entry={self.func.start.name}")
        for block in self.func.blocks():
            print(f"{block.name}:")
            block_info = self.info[id(block)]
            for i, inst in enumerate(block.instructions):
                types, values = block_info[i]
                print("  " + self.fmt_inst(inst, types, values))
            print()

    def optimize(self, env: EvalEnv) -> None:
        self.dataflow(env)
        self.merge_blocks()
        self.copy_propagate()
        self.remove_dead_code()

    def should_inline(
            self, func: SFunction, types: Optional[TypeTuple]
    ) -> bool:
        if func.name in self.banned_from_inline:
            return False
        name = func.name.name
        if name.startswith('inst/'):
            return True
        # @TODO: An actual inlining heuristic!
        SHOULD_INLINE = (
            'trap', 'trace', 'breakpoint', 'assert', 'typeof',
            'number?', 'symbol?', 'vector?', 'function?', 'bool?',
            'not',
            'pair?', 'nil?',
            'symbol=',
            # '+', '-', '*', '/', '%',
            # 'pointer=', 'number=', 'number<',
            # 'vector-length', 'vector-index', 'vector-set!',
            # '<', '!=', '>', '<=', '>=',
            # 'cons', 'car', 'cdr',
        )
        return name in SHOULD_INLINE
