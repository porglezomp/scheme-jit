echo "No optimizations"
make
echo "-tjbc0"
make FLAGS=-tjbc0
echo "-tjb"
make FLAGS=-tjb

echo "none, jc0, j, bc0, b, jbc0, jb"
make FLAGS=-jc0
make FLAGS=-j
make FLAGS=-bc0
make FLAGS=-b
make FLAGS=-jbc0
make FLAGS=-jb

echo "-jcN"
make FLAGS=-jc1
make FLAGS=-jc2
make FLAGS=-jc3
make FLAGS=-jc4
make FLAGS=-jc5
make FLAGS=-jc6
make FLAGS=-jc7
make FLAGS=-jc8
make FLAGS=-jc9
make FLAGS=-jc10

echo "-bcN"
make FLAGS=-jc1
make FLAGS=-jc2
make FLAGS=-jc3
make FLAGS=-jc4
make FLAGS=-jc5
make FLAGS=-jc6
make FLAGS=-jc7
make FLAGS=-jc8
make FLAGS=-jc9
make FLAGS=-jc10

make FLAGS=-bi10
make FLAGS=-bi20
make FLAGS=-bi30
make FLAGS=-bi40
make FLAGS=-bi50

touch "done"
