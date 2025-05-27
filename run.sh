python3 -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port --use_env run.py -c $1 -m $2
CUDA_VISIBLE_DEVICES=0 python run.py -c configs/tmad/tmad_256_100e.py -m train


conda install -c conda-forge accimage -y && conda install -c conda-forge faiss-cpu -y &&apt-get update&&apt-get install ffmpeg libsm6 libxext6 -y&& apt-get install libgl1 -y && cd /work/TMAD_ADer/ && pip install -r requirements.txt &&python -c "from opencv_fixer import AutoFix; AutoFix()" && cd dis_causal_conv1d && pip install -e . && cd .. &&cd dis_mamba && pip install -e . && cd ..  && CUDA_VISIBLE_DEVICES=0 python run.py -c configs/tmad/tmad_256_500e_mvtec.py -m train
