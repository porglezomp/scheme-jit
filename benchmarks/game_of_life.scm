(define (simulate rows cols steps live-cells)
  (simulate-helper rows cols steps
                   [(make-grid rows cols live-cells)
                           (make-grid rows cols '())
                   ]
  )
)

(define (simulate-helper rows cols steps grids)
  (if (> steps 0)
      (simulate-helper rows cols (- steps 1)
                       (simulate-step rows cols 1 grids)
      )
      0
      ;(vector-index grids 0) ; uncomment to return final grid
  )
)

(define (simulate-step rows cols current-row grids)
  (if (> current-row rows)
      (begin (print-grid rows cols (vector-index grids 1))
             [(vector-index grids 1) (vector-index grids 0)]
      )
      (begin (simulate-step-row rows cols current-row 1 grids)
             (simulate-step rows cols (+ current-row 1) grids)
      )
  )
)

(define (simulate-step-row rows cols current-row current-col grids)
  (if (> current-col cols)
      grids
      (begin (update-cell rows cols current-row current-col
                          (vector-index grids 0) (vector-index grids 1)
             )
             (simulate-step-row rows cols current-row (+ current-col 1) grids)
      )
  )
)

(define (update-cell rows cols row col old-grid new-grid)
  (grid-set new-grid rows cols row col
            (new-cell-value (sum [(grid-get old-grid rows cols (- row 1) (- col 1))
                                  (grid-get old-grid rows cols (- row 1) col)
                                  (grid-get old-grid rows cols (- row 1) (+ col 1))
                                  (grid-get old-grid rows cols row (- col 1))
                                  (grid-get old-grid rows cols row (+ col 1))
                                  (grid-get old-grid rows cols (+ row 1) (- col 1))
                                  (grid-get old-grid rows cols (+ row 1) col)
                                  (grid-get old-grid rows cols (+ row 1) (+ col 1))
                                  ]
                            )
                            (grid-get old-grid rows cols row col)
            )
  )
)

(define (sum vec)
  (sum-impl vec 0 0)
)

(define (sum-impl vec index total)
  (if (< index (vector-length vec))
    (sum-impl vec (+ index 1) (+ total (vector-index vec index)))
    total
  )
)

(define (new-cell-value neighbors old-value)
  (vector-index
    (vector-index [[0 0 0 1 0 0 0 0] [0 0 1 1 0 0 0 0]] old-value)
    neighbors)
)

(define (make-grid rows cols live-cells)
  (set-grid-cells (vector-make (* (+ rows 2) (+ cols 2)) 0)
                  rows cols live-cells
  )
)

(define (set-grid-cells grid rows cols live-cells)
  (if (nil? live-cells)
      grid
      (begin (grid-set grid rows cols (+ (caar live-cells) 1)
                       (+ (cadar live-cells) 1) 1
             )
             (set-grid-cells grid rows cols (cdr live-cells))
      )
  )
)

(define (caar p)
    (car (car p))
)

(define (cadar p)
    (car (cdr (car p)))
)

(define (grid-index rows cols padded-row padded-col)
  (+ (* padded-row (+ cols 2)) padded-col)
)

(define (grid-get grid rows cols padded-row padded-col)
  (vector-index grid (grid-index rows cols padded-row padded-col))
)

(define (grid-set grid rows cols padded-row padded-col value)
  (vector-set! grid (grid-index rows cols padded-row padded-col) value)
)

(define (print-grid rows cols grid)
  (print-header cols 0)
  (print-grid-helper rows cols 1 grid)
  (print-header cols 0)
  (newline)
)

(define (print-header cols index)
  (if (< index (+ cols 2))
      (begin (display 5)
             (print-header cols (+ index 1))
      )
      (newline)
  )
)

(define (print-grid-helper rows cols current-row grid)
  (if (<= current-row rows)
      (begin (display 5)
             (print-grid-row rows cols current-row 1 grid)
             (display 5)
             (newline)
             (print-grid-helper rows cols (+ current-row 1) grid)
      )
      0
  )
)

(define (print-grid-row rows cols current-row current-col grid)
  (if (<= current-col cols)
      (begin (display
                        (grid-get grid rows cols
                                    current-row current-col
                        )

             )
             (print-grid-row rows cols current-row (+ current-col 1) grid)
      )
      0
  )
)

(define (list vec) (list-impl vec 0 '()))

(define (list-impl vec index result)
    (if (number= index (vector-length vec))
        (reverse-tail result)
        (list-impl vec (+ index 1) (cons (vector-index vec index) result))
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


(define (life)
    (simulate 5 5 5
        (list [
            (list [0 3])
            (list [1 1])
            (list [1 3])
            (list [1 4])
            (list [2 3])
            (list [2 4])
            (list [3 0])
            (list [3 1])
            (list [3 2])
            (list [3 3])
            (list [4 0])
        ])
    )

;;;   (simulate 20 50 1
;;;         (list [
;;;             (list [1 25]) (list [2 23]) (list [2 25]) (list [3 13]) (list [3 14])
;;;             (list [3 21]) (list [3 22]) (list [3 35]) (list [3 36]) (list [4 12])
;;;             (list [4 16]) (list [4 21]) (list [4 22]) (list [4 35]) (list [4 36])
;;;             (list [5 1]) (list [5 2]) (list [5 11]) (list [5 17]) (list [5 21])
;;;             (list [5 22]) (list [6 1]) (list [6 2]) (list [6 11]) (list [6 15])
;;;             (list [6 17]) (list [6 18]) (list [6 23]) (list [6 25]) (list [7 11])
;;;             (list [7 17]) (list [7 25]) (list [8 12]) (list [8 16]) (list [9 13])
;;;             (list [9 14])
;;;         ])
;;;   )
)

(life)
