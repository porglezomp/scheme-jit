# Scheme JIT

Our goal is to demonstrate measurable benefits from some JIT optimizations.
We want tail call optimization so that we’ve got places to do loop optimization, so we'll have more visible improvements.
We want some monomorphization or something to get into an interesting JIT optimization.

## Values
- Numbers
- Symbols
- N-tuples (is `nil = []`?)
  Lists are nested pairs, so `'(1 2 3)` is `[1 [2 [3 []]]]`
  Immutable data
- Closures

## Monomorphization opportunities
- Specializing higher-order functions for a given function
- Removing bounds checks on tuple indexing
- Removing arity checks when calling functions
- Removing type checks before arithmetic / polymorphic comparison / etc.

## Special Forms

- `(define (f x ...) body ...)`
- `(if p t f)`
- `(quote x)`

## Builtin functions

- `(typeof x)`
- `(vector-make n x)`
- `(vector-index v n)`
- `(vector-set! v n x)`
- `(vector-length v)`
- `(pointer= a b)`
- `(number= a b)`
- `(symbol= a b)`
- `(symbol< a b)`

And arithmetic operators

## Prelude

- `=`, `!=`, `<`, `>`, `<=`, `>=`
- `not`
- `number?`, `symbol?`, `vector?`, `function?`, `pair?`, `nil?`
- `cons`, `car`, `cdr`

## Bytecode design

Split operations into many instructions.
For example, an `add_num` instruction wouldn't check the types of its arguments, typechecking would be separate.
Bad for efficiency, good for evaluating the effectiveness of JIT optimizations at removing bounds checks / type checks / etc.

We pick a register machine, in the hopes that it'll be easier to translate for optimizations.

## Sample code

```scheme
(define (nil? xs) (= xs []))
(define (car xs) (nth xs 0))
(define (cdr xs) (nth xs 1))
(define (cons x xs) [x xs])
(define (pair? x) (= (vlen x) 2))

(define (list? xs)
  (if (nil? xs)
    true
    (if (pair? xs)
      (list? (cdr xs))
      false)))

(define (map0 f xs)
  (if (nil? xs)
    []
    (cons
      (f (car xs))
      (map0 f (cdr xs)))))

(define (reverse-onto xs acc)
  (if (nil? xs)
    acc
    (reverse-onto
      (cdr xs)
      (cons (car xs) acc))))
      
(define (map-onto f xs acc)
  (if (nil? xs)
    acc
    (map-onto
      f (cdr xs)
      (cons (f (car xs) acc)))))
      
(define (map f xs)
  (reverse-onto
    (map-onto f xs [])
    []))
```
