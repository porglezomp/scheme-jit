(define (fib-tail n)
    (fib-tail-impl n 0 1)
)

(define (fib-tail-impl n first second)
    (if (= n 0)
        first
        (if (= n 1)
            second
            (fib-tail-impl (- n 1) second (+ first second))
        )
    )
)

(trace (fib-tail 0))
(trace (fib-tail 1))
(trace (fib-tail 2))
(trace (fib-tail 3))
(trace (fib-tail 4))
(trace (fib-tail 5))
(trace (fib-tail 6))
(trace (fib-tail 7))
(trace (fib-tail 8))
(trace (fib-tail 9))
'done
