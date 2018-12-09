import copy
from dataclasses import dataclass

import bytecode
from bytecode import BasicBlock, Function


@dataclass
class FunctionOptimizer:
    prefix_counter: int = 0

    def mark_vars(self, func: Function) -> Function:
        func = copy.deepcopy(func)
        prefix = f"inl{self.prefix_counter}"
        for block in func.blocks():
            block.name = f"{prefix}@{block.name}"
            assert isinstance(block, BasicBlock)
            for inst in block.instructions:
                inst.freshen(prefix)
        return func

    def inline_at_block_tail(self, block: BasicBlock, func: Function) -> None:
        assert len(block.instructions) >= 2
        assert isinstance(block.instructions[-2], bytecode.CallInst)
        assert isinstance(block.instructions[-1], bytecode.JmpInst)
        raise NotImplementedError("inlining")
