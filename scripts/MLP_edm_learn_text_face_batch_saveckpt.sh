#!/bin/bash
#SBATCH -t 20:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=75G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 8-9
#SBATCH -o MLP_edm_learn_saveckpt_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e MLP_edm_learn_saveckpt_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu


echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--dataset_name  ffhq-32x32                      --exp_name FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample                    --num_ckpts 100  --record_step_range 0 50000 500  --eval_sample_size 2048 --eval_batch_size 2048  --lr 1e-4 --nsteps 50000 --batch_size 256  --mlp_layers 8 --mlp_hidden_dim 3072 --mlp_time_embed_dim 128
--dataset_name  afhq-32x32                      --exp_name AFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample                     --num_ckpts 100  --record_step_range 0 50000 500  --eval_sample_size 2048 --eval_batch_size 2048  --lr 1e-4 --nsteps 50000 --batch_size 256  --mlp_layers 8 --mlp_hidden_dim 3072 --mlp_time_embed_dim 128
--dataset_name  CIFAR                           --exp_name CIFAR_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample                      --num_ckpts 100  --record_step_range 0 50000 500  --eval_sample_size 2048 --eval_batch_size 2048  --lr 1e-4 --nsteps 50000 --batch_size 256  --mlp_layers 8 --mlp_hidden_dim 3072 --mlp_time_embed_dim 128
--dataset_name  words32x32_50k                  --exp_name words32x32_50k_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample             --num_ckpts 100           --record_step_range 0 50000 500  --eval_sample_size 2048 --eval_batch_size 2048  --lr 1e-4 --nsteps 50000 --batch_size 256  --mlp_layers 8 --mlp_hidden_dim 3072 --mlp_time_embed_dim 128
--dataset_name  words32x32_50k_BW               --exp_name words32x32_50k_BW_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample          --num_ckpts 100              --record_step_range 0 50000 500  --eval_sample_size 2048 --eval_batch_size 2048  --lr 1e-4 --nsteps 50000 --batch_size 256  --mlp_layers 8 --mlp_hidden_dim 3072 --mlp_time_embed_dim 128
--dataset_name  ffhq-32x32-fix_words            --exp_name FFHQ32_fix_words_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample           --num_ckpts 100  --record_step_range 0 50000 500  --eval_sample_size 2048 --eval_batch_size 2048  --lr 1e-4 --nsteps 50000 --batch_size 256  --mlp_layers 8 --mlp_hidden_dim 3072 --mlp_time_embed_dim 128
--dataset_name  ffhq-32x32-random_word_jitter   --exp_name FFHQ32_random_words_jitter_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample --num_ckpts 100  --record_step_range 0 50000 500  --eval_sample_size 2048 --eval_batch_size 2048  --lr 1e-4 --nsteps 50000 --batch_size 256  --mlp_layers 8 --mlp_hidden_dim 3072 --mlp_time_embed_dim 128
--dataset_name  ffhq-32x32                      --exp_name FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample_longtrain           --num_ckpts 200  --record_step_range 0 250000 500  --eval_sample_size 2048 --eval_batch_size 2048  --lr 1e-4 --nsteps 250000 --batch_size 256  --mlp_layers 8 --mlp_hidden_dim 3072 --mlp_time_embed_dim 128
--dataset_name  words32x32_50k                  --exp_name words32x32_50k_UNet_MLP_EDM_8L_3072D_lr1e-4_saveckpt_fewsample_longtrain   --num_ckpts 200  --record_step_range 0 250000 500  --eval_sample_size 2048 --eval_batch_size 2048  --lr 1e-4 --nsteps 250000 --batch_size 256  --mlp_layers 8 --mlp_hidden_dim 3072 --mlp_time_embed_dim 128
'
# --dataset_name FFHQ                      --exp_name FFHQ_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
# --dataset_name  FFHQ_fix_words            --exp_name FFHQ_fix_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm             --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
# --dataset_name  FFHQ_random_words_jitter  --exp_name FFHQ_random_words_jitter_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm   --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 


export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

# load modules
module load python
mamba deactivate
mamba activate torch2
export PATH="$HOME/.conda/envs/torch2/bin:${PATH}"
which python

# run code
cd /n/home12/binxuwang/Github/DiffusionLearningCurve
python experiment/MLP_unet_learn_curve_CLI.py --record_frequency 0 --save_ckpts --eval_fix_noise_seed \
    $param_name

