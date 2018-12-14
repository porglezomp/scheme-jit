(define (assoc alist k)
  (if (nil? alist)
      []
      (if (= (car (car alist)) k)
          (cdr (car alist))
          (assoc (cdr alist) k))))

(define (to-bool x)
  (if (bool? x) x
  (if (number? x) (!= x 0)
  (if (nil? x) false
  true))))

(define (eval env s)
  ; if-chain as poor-man's cond
  (if (number? s) s
  (if (symbol? s) (assoc env s)
  (if (bool? s) s
  (if (nil? s) []
  (if (= (vector-length s) 1)
      ((eval env (vector-index s 0)))
  (if (= (vector-length s) 2)
      (if (= 'quote (vector-index s 0))
          (vector-index s 1)
          ((eval env (vector-index s 0))
           (eval env (vector-index s 1))))
  (if (= (vector-length s) 3)
      ((eval env (vector-index s 0))
       (eval env (vector-index s 1))
       (eval env (vector-index s 2)))
  (if (= (vector-length s) 4)
      ((eval env (vector-index s 0))
       (eval env (vector-index s 1))
       (eval env (vector-index s 2))
       (eval env (vector-index s 3)))
  (trap))))))))))

((lambda (env)
   (trace (eval env
                ['? ['= ['+ ['- 'a 'b]
                            ['- 'b 'a]]
                        0]
                    ['quote '(h e l l o)]
                    ['* 420 666]]))
   (trace (eval env
                ['? ['print ['< ['print
                                 ['- ['print
                                      ['+ ['* 'a 'b]
                                          ['* 'a 'b]]]
                                     ['print
                                      ['+ ['* 'a 'a]
                                          ['* 'b 'b]]]]]
                                0]]
                    ['- 0 ['- ['+ ['* 'a 'b]
                                  ['* 'a 'b]]
                              ['+ ['* 'a 'a]
                                  ['* 'b' b]]]]
                    ['- ['+ ['* 'a 'b]
                            ['* 'a 'b]]
                        ['+ ['* 'a 'a]
                            ['* 'b' b]]]]))
   'done
   ) (list [['+ +]
            ['- -]
            ['* *]
            ['/ *]
            ['< <]
            ['= =]
            ['print trace]
            ['? (lambda (b x y) (if (to-bool b) x y))]
            ['a 42]
            ['b 69]]))
