cfg=$1
batch_size=64

state_epoch=1 
pretrained_model_path='./saved_models/data/model_save_file'
log_dir='new'

multi_gpus=False
mixed_precision=True
gpu_id=0



nodes=1
num_workers=8
master_port=11266
stamp=gpu${nodes}MP_${mixed_precision}

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$nodes --master_port=$master_port src/train.py \
                    --stamp $stamp \
                    --cfg $cfg \
                    --mixed_precision $mixed_precision \
                    --log_dir $log_dir \
                    --num_workers $num_workers \