nohup python train_parallel.py -env $1 -gpu > out.log 2>&1 &
echo $! >pid.txt