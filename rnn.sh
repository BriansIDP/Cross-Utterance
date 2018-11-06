export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
export PATH="/home/miproj/urop.2018/gs534/Software/anaconda3/bin:$PATH"

python main.py --data data/penn-treebank/ --cuda --emsize 256 --nhid 256 --dropout 0.5 --epochs 30 --lr 0.001 --nlayers 1 --batch_size 64 --bptt 30 --model LSTM
python L2LMinput.py --data data/penn-treebank/ --cuda --model model.pt --memorycell --uttlookback 3 --saveprefix tensors/penn-treebank 
