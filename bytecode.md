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
- `store <to> = <from>`

## Branching
- `jmp block`
- `br value block_t block_f`

## Call
- `calli value (args)`
- (later? `call name (args)`)

## Sample code
### Recursive version of (list?)
```
function list? (v0) entry=bb0
bb0:
  result = alloc

  v1 = global nil?
  v2 = call v1 (v0)
  br v2 bb1 bb2

bb1:
  v10 = sym true
  store result = v10
  jmp end

bb2:
  v3 = global pair?
  v4 = call v3 (v0)
  br v4 bb3 bb4

bb3:
  v5 = global cdr
  v6 = call v5 (v0)
  v7 = global list?
  v8 = call v7 (v6)
  store result = v8
  jmp end

bb4:
  v9 = sym false
  store result = v9
  jmp end

end:
  return result
```

### Tail-call-optimized version of (list?)
```
function list? (v0) entry=bb0
bb0:
  result = alloc
  so_far = alloc
  store so_far = v0

bb00:
  v11 = load so_far

  v1 = global nil?
  v2 = call v1 (v11)
  br v2 bb1 bb2

bb1:
  v10 = sym true
  store result = v10
  jmp end

bb2:
  v3 = global pair?
  v4 = call v3 (v0)
  br v4 bb3 bb4

bb3:
  v5 = global cdr
  v6 = call v5 (v0)
  store so_far = v6
  jmp bb00

bb4:
  v9 = sym false
  store result = v9
  jmp end

end:
  return result
```
