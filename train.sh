while :
do
    OMP_NUM_THREADS=16 python train.py ;
    if [ $? -eq 2 ]
    then
        break
    fi
done