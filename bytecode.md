# Bytecode
A function is a control flow graph that begins at an entry block.
The blocks are extended basic blocks, which can have multiple exit points, but will always be entered at the top.
Every block must be terminated by a `jmp`, `return`, or `trap` instruction, there is no fallthrough.

Inspired by LLVM IR

# Instructions

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

## Copy
- `copy`

## Memory
- `alloc`
- `load`
- `store <to> = <from>`
- `lookup`
- `length`

## Branching
- `jmp block`
- `br value block`
- `brn value block`

## Call
- `call value (args)`
- `return`

## Error
- `trap`

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
