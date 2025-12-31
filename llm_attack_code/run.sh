export MASTER_PORT=$((12000 + $RANDOM % 20000))

torchrun --nproc-per-node 8 --master_port=$MASTER_PORT run.py --model=qwen --bad_word=0
