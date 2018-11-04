export CUDA_VISIBLE_DEVICES=1 
python L2LMinput.py --data data/AMI/ --cuda --model model.pt --memorycell --uttlookback 3 --batchsize 3
