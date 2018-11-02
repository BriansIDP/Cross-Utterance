rm LOGs/log
qsub -cwd -S /bin/bash -l qp=cuda-low,tests=0,mem_grab=0M,osrel="*",gpuclass="*" -o LOGs/log -j y rnn.sh
