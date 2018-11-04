export CUDA_VISIBLE_DEVICES=1 
python main.py --data smalldata/ --cuda --emsize 3 --nhid 3 --dropout 0.5 --epochs 1 --lr 0.001 --nlayers 1 --batch_size 2 --bptt 5 --model LSTM --reset 0
python L2LMinput.py --data smalldata/ --cuda --model model.pt --memorycell --uttlookback 2 --batchsize 3
