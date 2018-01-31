for i in $(seq 0 17 | shuf); do
    python kks.py $i
done