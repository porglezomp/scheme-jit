# Usage examples:
#
# Runs the "fib_tail.scm" benchmark with no optimizations:
# 	make fib_tail.bench
#
# Runs all benchmarks with all optimizations:
# 	make FLAGS=-tjb
#
# Data files are written to data/
#

FLAGS =

benchmark_files := $(notdir $(wildcard benchmarks/*.scm))
correct_files := $(benchmark_files:.scm=.correct)
benchmark_targets := $(benchmark_files:.scm=.bench)

all: $(benchmark_targets)

%.bench:
	@echo $@
	@mkdir -p data
	./scheme.py $(FLAGS) \
		-fsa -m -o data/$*_$(FLAGS).json benchmarks/$*.scm > benchmarks/$*.out
	@diff -q benchmarks/$*.correct benchmarks/$*.out

.PHONY: all clean

clean:
	rm benchmarks/*.out
