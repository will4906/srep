for i in $(seq 0 17 | shuf); do
    echo 'starting' $i
    python kks.py $i
done