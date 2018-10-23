# Scheme JIT

Our goal is to demonstrate measurable benefits from some JIT optimizations.
We want tail call optimization so that we’ve got places to do loop optimization, so we'll have more visible improvements.
We want some monomorphization or something to get into an interesting JIT optimization.

## Values
- Numbers
- Symbols
- N-tuples (is `nil = []`?)
  Lists are nested pairs, so `'(1 2 3)` is `[1 [2 [3 []]]]`
- Closures

## Monomorphization opportunities
- Specializing higher-order functions for a given function
- Removing bounds checks on tuple indexing
- Removing arity checks when calling functions
- Removing type checks before arithmetic / polymorphic comparison / etc.

