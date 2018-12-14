(define (calc inputs)
    (calc-impl inputs (dlist-make))
)

(define (calc-impl inputs stack)
    (if (nil? inputs)
        (trace 'calc-stack-empty)
        (begin
            ((lambda (stack next-input)
                (if (number? next-input)
                    (dlist-push-front stack next-input)
                    (if (= next-input 'p)  ; print
                        (dlist-print stack)
                        (if (= next-input 'd)  ; duplicate top elt
                            (dlist-push-front stack (dlist-front stack))
                            (if (= next-input 'n)  ; negate top elt
                                (dlist-push-front stack (- 0 (dlist-pop-front stack)))
                                (if (= next-input '+)
                                    (dlist-push-front stack (+ (dlist-pop-front stack)
                                                            (dlist-pop-front stack))
                                    )
                                    (if (= next-input '-)
                                        (dlist-push-front stack (- (dlist-pop-front stack)
                                                                (dlist-pop-front stack))
                                        )
                                        (if (= next-input '*)
                                            (dlist-push-front stack (* (dlist-pop-front stack)
                                                                        (dlist-pop-front stack))
                                            )
                                            (if (= next-input '/)
                                                (dlist-push-front stack (/ (dlist-pop-front stack)
                                                                        (dlist-pop-front stack))
                                                )
                                                (trace 'bad-input)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ) stack (car inputs))
            (calc-impl (cdr inputs) stack)
        )
    )
)

; ----------------------------------------------------------------------

; Makes a doubly-linked list
(define (dlist-make)
    [0 0]
)

(define (dlist-empty dlist)
    (= (dlist-first dlist) 0)
)

(define (dlist-push-front dlist datum)
    (if (dlist-empty dlist)
        ((lambda (dlist new-node)
            (dlist-set-first! dlist new-node)
            (dlist-set-last! dlist new-node)
        ) dlist (node-make datum 0 0))

        ((lambda (dlist current-first new-node)
            (node-set-prev! current-first new-node)
            (dlist-set-first! dlist new-node)
        ) dlist (dlist-first dlist) (node-make datum 0 (dlist-first dlist)))
    )
)

(define (dlist-push-back dlist datum)
    (if (dlist-empty dlist)
        ((lambda (dlist new-node)
            (dlist-set-first! dlist new-node)
            (dlist-set-last! dlist new-node)
        ) dlist (node-make datum 0 0))

        ((lambda (dlist current-last new-node)
            (node-set-next! current-last new-node)
            (dlist-set-last! dlist new-node)
        ) dlist (dlist-last dlist) (node-make datum (dlist-last dlist) 0))
    )
)

(define (dlist-front dlist)
    (assert (not (dlist-empty dlist)))
    (node-get-datum (dlist-first dlist))
)

(define (dlist-back dlist)
    (assert (not (dlist-empty dlist)))
    (node-get-datum (dlist-last dlist))
)

(define (dlist-pop-front dlist)
    (assert (not (dlist-empty dlist)))
    (if (pointer= (dlist-first dlist) (dlist-last dlist))
        (begin
            ((lambda (dlist first)
                (dlist-set-first! dlist 0)
                (dlist-set-last! dlist 0)
                (node-get-datum first)
            ) dlist (dlist-first dlist))
        )
        ((lambda (dlist current-first new-first)
            (node-set-prev! new-first 0)
            (node-set-next! current-first 0)
            (dlist-set-first! dlist new-first)
            (node-get-datum current-first)
        ) dlist (dlist-first dlist) (node-get-next (dlist-first dlist)))
    )
)

(define (dlist-pop-back dlist)
    (assert (not (dlist-empty dlist)))
    (if (pointer= (dlist-first dlist) (dlist-last dlist))
        (begin
            ((lambda (dlist first)
                (dlist-set-first! dlist 0)
                (dlist-set-last! dlist 0)
                (node-get-datum first)
            ) dlist (dlist-first dlist))
        )
        ((lambda (dlist current-last new-last)
            (node-set-next! new-last 0)
            (node-set-prev! current-last 0)
            (dlist-set-last! dlist new-last)
            (node-get-datum current-last)
        ) dlist (dlist-last dlist) (node-get-prev (dlist-last dlist)))
    )
)

(define (dlist-clear dlist)
    (dlist-set-first! dlist 0)
    (dlist-set-last! dlist 0)
)

(define (dlist-print dlist)
    (dlist-print-impl (dlist-first dlist))
)

(define (dlist-print-impl node)
    (if (= node 0)
        (trace 'end_of_list)
        (begin
            (trace (node-get-datum node))
            (dlist-print-impl (node-get-next node))
        )
    )
)

; ----------------------------------------------------------------------

(define (dlist-first dlist)
    (vector-index dlist 0)
)

(define (dlist-set-first! dlist new-first)
    (vector-set! dlist 0 new-first)
)

(define (dlist-last dlist)
    (vector-index dlist 1)
)

(define (dlist-set-last! dlist new-last)
    (vector-set! dlist 1 new-last)
)

; ----------------------------------------------------------------------

(define (node-make datum prev next)
    [datum prev next]
)

(define (node-get-datum node)
    (vector-index node 0)
)

(define (node-get-prev node)
    (vector-index node 1)
)

(define (node-set-prev! node new_prev)
    (vector-set! node 1 new_prev)
)

(define (node-get-next node)
    (vector-index node 2)
)

(define (node-set-next! node new_next)
    (vector-set! node 2 new_next)
)

; ----------------------------------------------------------------------

(define (and a b)
    (if a b a)
)

(define (or a b)
    (if a a b)
)

; ----------------------------------------------------------------------

; Some tests for the dlist

(define (test-front dlist)
    (dlist-push-front dlist 1)
    (dlist-push-front dlist 2)
    (dlist-push-front dlist 3)
    (dlist-push-front dlist 4)
    (dlist-push-front dlist 5)
    (dlist-push-front dlist 6)
    (dlist-push-front dlist 7)
    (dlist-push-front dlist 8)
    (dlist-print dlist)

    (trace (dlist-pop-front dlist))
    (trace (dlist-pop-front dlist))
    (trace (dlist-pop-front dlist))
    (trace (dlist-pop-front dlist))
    (trace (dlist-pop-front dlist))
    (trace (dlist-pop-front dlist))
    (trace (dlist-pop-front dlist))
    (trace (dlist-pop-front dlist))

    (dlist-print dlist)
)

(test-front (dlist-make))

(define (test-back dlist)
    (dlist-push-back dlist 1)
    (dlist-push-back dlist 2)
    (dlist-push-back dlist 3)
    (dlist-push-back dlist 4)
    (dlist-push-back dlist 5)
    (dlist-push-back dlist 6)
    (dlist-push-back dlist 7)
    (dlist-push-back dlist 8)
    (dlist-print dlist)

    (trace (dlist-pop-back dlist))
    (trace (dlist-pop-back dlist))
    (trace (dlist-pop-back dlist))
    (trace (dlist-pop-back dlist))
    (trace (dlist-pop-back dlist))
    (trace (dlist-pop-back dlist))
    (trace (dlist-pop-back dlist))
    (trace (dlist-pop-back dlist))

    (dlist-print dlist)
)

(test-back (dlist-make))

(calc '(1 p 2 p + 3 p * p 4 p 5 p))

(calc '(
    62 p 10 p d p - p n p d p n p 93 p * p n
))


(calc '(
    31 p 63 p n p 15 p + p 87 p + p d p d p * p 20 p * p 14 p 55 p n p 67 p d p n p n p *
))

(calc '(
    82 p n p 32 p d p 42 p + p * p 20 p 63 p 32 p 5 p n p 22 p 21 p n p - p 65 p + p 67 p
    * p * p - p 1 p - p 88 p n p 37 p - p - p *
))

;;; These take a long time without any optimizations
;;; (calc '(
;;;     41 p 17 p 57 p 25 p 20 p 18 p 56 p 62 p 86 p 10 p 91 p 69 p * p 83 p 28 p - p 90 p 46
;;;     p * p - p * p 78 p 77 p 43 p d p 11 p 42 p 20 p 98 p 64 p + p * p 36 p 41 p d p 52 p
;;;     90 p 61 p n p 56
;;; ))

;;; (calc '(
;;;     9 p n p 44 p + p 94 p 84 p 99 p 43 p + p 79 p - p d p 34 p * p 36 p n p 23 p 28 p * p
;;;     100 p - p 1 p * p 24 p 71 p + p - p + p + p d p n p * p 2 p 56 p 1 p + p 89 p + p * p
;;;     + p * p 28 p 31 p * p 90 p 65 p d p 8 p * p +
;;; ))
