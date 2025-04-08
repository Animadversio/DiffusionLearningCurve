#!/bin/bash
#SBATCH -t 12:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=75G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array=4
#SBATCH -o DiT_edm_learn_curve_saveckpt_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e DiT_edm_learn_curve_saveckpt_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--dataset_name ffhq-32x32                     --exp_name FFHQ32_DiT_P2_192D_3H_6L_EDM_saveckpt_fewsample    --patch_size 2 --hidden_size 192 --depth 6 --num_heads 3 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
--dataset_name afhq-32x32                      --exp_name AFHQ32_DiT_P2_192D_3H_6L_EDM_saveckpt_fewsample    --patch_size 2 --hidden_size 192 --depth 6 --num_heads 3 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
--dataset_name CIFAR                           --exp_name CIFAR_DiT_P2_192D_3H_6L_EDM_saveckpt_fewsample     --patch_size 2 --hidden_size 192 --depth 6 --num_heads 3 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
--dataset_name words32x32_50k                  --exp_name words32x32_50k_DiT_P2_192D_3H_6L_EDM_saveckpt_fewsample     --patch_size 2 --hidden_size 192 --depth 6 --num_heads 3 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
--dataset_name ffhq-32x32                     --exp_name FFHQ32_DiT_P4_384D_6H_6L_EDM_saveckpt_fewsample    --patch_size 4 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
--dataset_name afhq-32x32                      --exp_name AFHQ32_DiT_P4_384D_6H_6L_EDM_saveckpt_fewsample    --patch_size 4 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
--dataset_name CIFAR                           --exp_name CIFAR_DiT_P4_384D_6H_6L_EDM_saveckpt_fewsample     --patch_size 4 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
'
# --dataset_name ffhq-32x32                      --exp_name FFHQ32_DiT_P2_384D_6H_6L_EDM_pilot    --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name ffhq-32x32                      --exp_name FFHQ32_DiT_P4_384D_6H_6L_EDM_pilot    --patch_size 4 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name ffhq-32x32                      --exp_name FFHQ32_DiT_P2_768D_12H_6L_EDM_pilot   --patch_size 2 --hidden_size 768 --depth 6 --num_heads 12 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name ffhq-32x32                      --exp_name FFHQ32_DiT_P2_768D_12H_12L_EDM_pilot  --patch_size 2 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name afhq-32x32                      --exp_name AFHQ32_DiT_P2_192D_3H_6L_EDM_pilot    --patch_size 2 --hidden_size 192 --depth 6 --num_heads 3 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name CIFAR                           --exp_name CIFAR_DiT_P2_192D_3H_6L_EDM_pilot     --patch_size 2 --hidden_size 192 --depth 6 --num_heads 3 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name ffhq-32x32                      --exp_name FFHQ32_DiT_P4_768D_12H_6L_EDM_pilot  --patch_size 4 --hidden_size 768 --depth 6 --num_heads 12 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name ffhq-32x32                      --exp_name FFHQ32_DiT_P4_768D_12H_12L_EDM_pilot --patch_size 4 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name words32x32_50k                  --exp_name words32x32_50k_DiT_P2_384D_6H_6L_EDM_pilot  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name words32x32_50k                  --exp_name words32x32_50k_DiT_P4_768D_12H_12L_EDM_pilot --patch_size 4 --hidden_size 768 --depth 12 --num_heads 12 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024  --lr 1e-4 --nsteps 50000 --batch_size 256  

# --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 
# --dataset_name  FFHQ_fix_words            --exp_name FFHQ_fix_words_DiT_EDM_pilot            --eval_sample_size 2000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name  FFHQ_random_words_jitter  --exp_name FFHQ_random_words_jitter_DiT_EDM_pilot  --eval_sample_size 2000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name  ffhq-32x32                      --exp_name FFHQ32_DiT_EDM_pilot                      --eval_sample_size 2000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name  ffhq-32x32-fix_words            --exp_name FFHQ32_fix_words_DiT_EDM_pilot            --eval_sample_size 2000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name  ffhq-32x32-random_word_jitter   --exp_name FFHQ32_random_words_jitter_DiT_EDM_pilot  --eval_sample_size 2000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name  afhq-32x32                      --exp_name AFHQ32_DiT_EDM_pilot                      --eval_sample_size 2000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  
# --dataset_name  CIFAR                           --exp_name CIFAR_DiT_EDM_pilot                       --eval_sample_size 2000 --eval_batch_size 512  --lr 1e-4 --nsteps 50000 --batch_size 256  


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
python experiment/DiT_learn_curve_CLI.py --record_frequency 0 --eval_sampling_steps 35  --save_ckpts --num_ckpts 100 --eval_fix_noise_seed  --record_step_range 0 50000 500   \
    $param_name

