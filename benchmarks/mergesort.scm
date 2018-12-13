(define (mergesort vec)
    (if (number= (vector-length vec) 1)
        vec
        (slice-and-mergesort
            vec
            (/ (vector-length vec) 2)
            (vector-length vec)
        )
    )

)

(define (slice-and-mergesort vec middle end)
    (merge
        (mergesort (slice vec 0 middle))
        (mergesort (slice vec middle end))
    )
)


(define (merge first second)
    (merge-impl
        first 0 second 0
        (vector-make (+ (vector-length first) (vector-length second)) 0)
        0
    )
)

(define (merge-impl first first_index second second_index result result_index)
    (if (and (< first_index (vector-length first))
             (< second_index (vector-length second)))
        (if (<= (vector-index first first_index) (vector-index second second_index))
            (begin
                (vector-set! result result_index (vector-index first first_index))
                (merge-impl first (+ first_index 1)
                            second second_index
                            result (+ result_index 1))
            )
            (begin
                (vector-set! result result_index (vector-index second second_index))
                (merge-impl first first_index
                            second (+ second_index 1)
                            result (+ result_index 1))
            )
        )

        (if (< first_index (vector-length first))
            (begin

                (vector-set! result result_index (vector-index first first_index))
                (merge-impl first (+ first_index 1)
                            second second_index
                            result (+ result_index 1))
            )
            (if (< second_index (vector-length second))
                (begin
                    (vector-set! result result_index (vector-index second second_index))
                    (merge-impl first first_index
                                second (+ second_index 1)
                                result (+ result_index 1))
                )
                result
            )
        )
    )
)

(define (slice vec start end)
    (copy-range vec (vector-make (- end start) 0) start end)
)

(define (copy-range from to start end)
    (copy-range-impl from to start end 0)
)

(define (copy-range-impl from to start end to_index)
    (if (= start end)
        to
        (begin
            (vector-set! to to_index (vector-index from start))
            (copy-range-impl from to (+ start 1) end (+ to_index 1))
        )
    )
)

(define (empty? vec)
    (number= (vector-length vec) 0)
)

(define (and a b)
    (if a b a)
)

(define (or a b)
    (if a a b)
)

; Tests for the pieces
;(trace (merge [1 3 5] [2 4]))
;(trace (merge [2 4] [1 3 5]))
;(trace (merge [1 2 3 5 5 7] [3 5]))

;(trace (copy-range [1 2 3 4] (vector-make 2 0) 2 4))
;(trace (slice [1 2 3 4] 2 4))
;(trace (slice [1 2 3 4] 1 3))
;(trace (slice [1 2 3 4] 0 4))

;(trace (slice [1 2 3 4 5 6 7 8 9 10] 0 10))

(trace (mergesort [-9 -9 6 6 7 8 -2 0 6 5]))
(trace (mergesort [9 10 -2 -6 10 -3 5 -9 -8 -10 9 -7 3 5 8 -6 -10 -4 -9 -8]))
(trace (mergesort [22 17 -16 22 14 -20 -20 -4 21 -8 9 -20 10 16 -13 14 18 -9 -8 -19 -6 7 4 16 11 -25 -4 6 -1 -10 2 -21 -23 -19 -23 -17 6 16 1 -7 -15 -22 1 3 13 4 -15 -10 6 25]))
'done

; Use this to generate calls to mergesort
;def mergy(width, num_elts):
;    def rand():
;        return random.randint(-(width // 2), width // 2)
;
;    def nums():
;        return ' '.join((str(rand()) for i in range(num_elts)))
;    return f'(trace (mergesort [{nums()}]))'
