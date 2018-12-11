(define (map-tail list func)
    (map-tail-impl list func '())
)

(define (map-tail-impl list func result)
    (if (nil? list)
        (reverse-tail result)
        (map-tail-impl (cdr list) func (cons (func (car list)) result) )
    )
)


(define (reverse-tail list)
    (reverse-tail-impl list '())
)

(define (reverse-tail-impl list result)
    (if (nil? list)
        result
        (reverse-tail-impl (cdr list) (cons (car list) result))
    )
)

(map-tail '(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15) (lambda (elt) (+ elt 1)))
