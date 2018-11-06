export CUDA_VISIBLE_DEVICES=1
python trainL2.py --cuda --seed 1111 --nhid 256 --emsize 256 --lr 0.001 --batchsize 64 --bptt 30 --naux 64
