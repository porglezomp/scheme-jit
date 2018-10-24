# Bytecode
A function is a list of basic blocks, and entry block, and an exit block
Each basic block has a terminator instruction which says which next block to branch to
The final block is just the return instruction

Inspired by LLVM IR

# Instructions

## Literals
- `num`
- `sym`

## Arithmetic
- `add`
- `sub`
- `mul`
- `div`
- `mod`

## Comparison
- `num_eq`
- `sym_eq`
- `ptr_eq`
- `num_lt`

## Type extraction
- `typeof`

## Memory
- `alloc`
- `load`
- `store`

## Branching
- `jmp block`
- `br value block_t block_f`

## Call
- `calli value (args)`
- (later? `call name (args)`)


```
function list? (v0) entry=bb0
bb0:
  v1 = global nil?
  v2 = call v1 (v0)
  br v2 bb1 bb2
  
bb1:
  result = sym true
  jmp end

bb2:

end:
  return
  
```
