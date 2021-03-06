(define (fib n)
    (if (= n 0)
        0
        (if (= n 1)
            1
            (+ (fib (- n 1)) (fib (- n 2)))
        )
    )
)

(trace (fib 0))
(trace (fib 1))
(trace (fib 2))
(trace (fib 3))
(trace (fib 4))
(trace (fib 5))
(trace (fib 6))
(trace (fib 7))
(trace (fib 8))
(trace (fib 9))
'done
