CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --resume_from_checkpoint logs/version_4/epoch=55-val_loss=0.046.ckpt \
  --max_epochs 400 --warmup_ratio 0.05 \
  --batch_size 80 --max_len 200 \
  --num_workers 8 --lr 5e-4 \
  --default_root_dir logs  --gpus 2 \
  --train_file data/train --valid_file data/valid 