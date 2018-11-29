(define (reverse-tail list)
    (reverse-tail-impl list '())
)

(define (reverse-tail-impl list result)
    (if (nil? list)
        result
        (reverse-tail-impl (cdr list) (cons (car list) result))
    )
)

(trace (reverse-tail '(1 2)))
(trace (reverse-tail '(1 2 3)))
(trace (reverse-tail '(1 2 3 4)))
(trace (reverse-tail '(1 2 3 4 5)))
(trace (reverse-tail '(1 2 3 4 5 6)))
