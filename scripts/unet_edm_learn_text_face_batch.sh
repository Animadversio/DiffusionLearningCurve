#!/bin/bash
#SBATCH -t 10:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=75G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 19-21
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o unet_edm_learn_text_face_ESM_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e unet_edm_learn_text_face_ESM_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--dataset_name FFHQ                      --exp_name FFHQ_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
--dataset_name  FFHQ_fix_words            --exp_name FFHQ_fix_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm             --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
--dataset_name  FFHQ_random_words_jitter  --exp_name FFHQ_random_words_jitter_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm   --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
--dataset_name  ffhq-32x32                      --exp_name FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
--dataset_name  ffhq-32x32-fix_words            --exp_name FFHQ32_fix_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm             --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
--dataset_name  ffhq-32x32-random_word_jitter   --exp_name FFHQ32_random_words_jitter_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm   --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
--dataset_name  afhq-32x32                      --exp_name AFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
--dataset_name  CIFAR                           --exp_name CIFAR_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm                        --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
--dataset_name  ffhq-32x32                      --exp_name FFHQ32_UNet_CNN_EDM_3blocks_2x_wide128_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 --attn_resolutions 0  --layers_per_block 2
--dataset_name  ffhq-32x32                      --exp_name FFHQ32_UNet_CNN_EDM_2blocks_2x_wide128_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 --attn_resolutions 0  --layers_per_block 2
--dataset_name  ffhq-32x32                      --exp_name FFHQ32_UNet_CNN_EDM_1blocks_2x_wide128_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 --attn_resolutions 0    --layers_per_block 2 
--dataset_name  afhq-32x32                      --exp_name AFHQ32_UNet_CNN_EDM_3blocks_2x_wide128_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 --attn_resolutions 0  --layers_per_block 2
--dataset_name  afhq-32x32                      --exp_name AFHQ32_UNet_CNN_EDM_2blocks_2x_wide128_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 --attn_resolutions 0  --layers_per_block 2
--dataset_name  afhq-32x32                      --exp_name AFHQ32_UNet_CNN_EDM_1blocks_2x_wide128_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 --attn_resolutions 0    --layers_per_block 2 
--dataset_name  afhq-32x32                      --exp_name AFHQ32_UNet_CNN_EDM_1blocks_1x_wide128_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 --attn_resolutions 0    --layers_per_block 1  --eval_fix_noise_seed
--dataset_name  ffhq-32x32                      --exp_name FFHQ32_UNet_CNN_EDM_1blocks_1x_wide128_fixednorm                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 --attn_resolutions 0    --layers_per_block 1  --eval_fix_noise_seed
--dataset_name  ffhq-32x32     --loss_type ESM  --exp_name FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm_ESM                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8    --eval_fix_noise_seed
--dataset_name  ffhq-32x32     --loss_type DSM  --exp_name FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm_DSM                       --decoder_init_attn True  --eval_sample_size 1000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8    --eval_fix_noise_seed
--dataset_name  ffhq-32x32     --loss_type DSM  --exp_name FFHQ32_UNet_CNN_EDM_3blocks_1x_wide8_pilot_fixednorm                      --eval_sample_size 2000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 8 --channel_mult 1 2 2 --attn_resolutions 0  --layers_per_block 1 --eval_fix_noise_seed
--dataset_name  ffhq-32x32     --loss_type DSM  --exp_name FFHQ32_UNet_CNN_EDM_2blocks_1x_wide8_pilot_fixednorm                      --eval_sample_size 2000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 8 --channel_mult 1 2   --attn_resolutions 0  --layers_per_block 1 --eval_fix_noise_seed
--dataset_name  ffhq-32x32     --loss_type DSM  --exp_name FFHQ32_UNet_CNN_EDM_1blocks_1x_wide8_pilot_fixednorm                      --eval_sample_size 2000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  --model_channels 8 --channel_mult 1     --attn_resolutions 0  --layers_per_block 1 --eval_fix_noise_seed
'

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
python experiment/CNN_unet_learn_curve_CLI.py --record_frequency 0 \
    $param_name

