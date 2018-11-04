export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
export PATH="/home/miproj/urop.2018/gs534/Software/anaconda3/bin:$PATH"

python main.py --data data/AMI/ --cuda --emsize 256 --nhid 256 --dropout 0.5 --epochs 3 --lr 0.001 --nlayers 1 --batch_size 64 --bptt 24 --model LSTM
python L2LMinput.py --data data/AMI/ --cuda --model model.pt --memorycell --uttlookback 3 --batchsize 3
