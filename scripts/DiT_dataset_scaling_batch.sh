#!/bin/bash
#SBATCH -t 16:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=75G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array=1-10
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o DiT_dataset_scaling_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e DiT_dataset_scaling_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--dataset_name ffhq-32x32   --dset_start     0 --dset_end   300 --loss_type DSM --exp_name FFHQ32_300_DiT_P2_384D_6H_6L_EDM_DSM_split1   --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name ffhq-32x32   --dset_start   300 --dset_end   600 --loss_type DSM --exp_name FFHQ32_300_DiT_P2_384D_6H_6L_EDM_DSM_split2   --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name ffhq-32x32   --dset_start     0 --dset_end  1000 --loss_type DSM --exp_name FFHQ32_1000_DiT_P2_384D_6H_6L_EDM_DSM_split1  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name ffhq-32x32   --dset_start  1000 --dset_end  2000 --loss_type DSM --exp_name FFHQ32_1000_DiT_P2_384D_6H_6L_EDM_DSM_split2  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name ffhq-32x32   --dset_start     0 --dset_end  3000 --loss_type DSM --exp_name FFHQ32_3000_DiT_P2_384D_6H_6L_EDM_DSM_split1  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name ffhq-32x32   --dset_start  3000 --dset_end  6000 --loss_type DSM --exp_name FFHQ32_3000_DiT_P2_384D_6H_6L_EDM_DSM_split2  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name ffhq-32x32   --dset_start     0 --dset_end 10000 --loss_type DSM --exp_name FFHQ32_10000_DiT_P2_384D_6H_6L_EDM_DSM_split1 --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name ffhq-32x32   --dset_start 10000 --dset_end 20000 --loss_type DSM --exp_name FFHQ32_10000_DiT_P2_384D_6H_6L_EDM_DSM_split2 --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name ffhq-32x32   --dset_start     0 --dset_end 30000 --loss_type DSM --exp_name FFHQ32_30000_DiT_P2_384D_6H_6L_EDM_DSM_split1 --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name ffhq-32x32   --dset_start 30000 --dset_end 60000 --loss_type DSM --exp_name FFHQ32_30000_DiT_P2_384D_6H_6L_EDM_DSM_split2 --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name afhq-32x32   --dset_start     0 --dset_end   300 --loss_type DSM --exp_name AFHQ32_300_DiT_P2_384D_6H_6L_EDM_DSM_split1   --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name afhq-32x32   --dset_start   300 --dset_end   600 --loss_type DSM --exp_name AFHQ32_300_DiT_P2_384D_6H_6L_EDM_DSM_split2   --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name afhq-32x32   --dset_start     0 --dset_end  1000 --loss_type DSM --exp_name AFHQ32_1000_DiT_P2_384D_6H_6L_EDM_DSM_split1  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name afhq-32x32   --dset_start  1000 --dset_end  2000 --loss_type DSM --exp_name AFHQ32_1000_DiT_P2_384D_6H_6L_EDM_DSM_split2  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name afhq-32x32   --dset_start     0 --dset_end  3000 --loss_type DSM --exp_name AFHQ32_3000_DiT_P2_384D_6H_6L_EDM_DSM_split1  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name afhq-32x32   --dset_start  3000 --dset_end  6000 --loss_type DSM --exp_name AFHQ32_3000_DiT_P2_384D_6H_6L_EDM_DSM_split2  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name afhq-32x32   --dset_start     0 --dset_end 10000 --loss_type DSM --exp_name AFHQ32_10000_DiT_P2_384D_6H_6L_EDM_DSM_split1 --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name afhq-32x32   --dset_start 10000 --dset_end 20000 --loss_type DSM --exp_name AFHQ32_10000_DiT_P2_384D_6H_6L_EDM_DSM_split2 --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name afhq-32x32   --dset_start     0 --dset_end 30000 --loss_type DSM --exp_name AFHQ32_30000_DiT_P2_384D_6H_6L_EDM_DSM_split1 --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name afhq-32x32   --dset_start 30000 --dset_end 60000 --loss_type DSM --exp_name AFHQ32_30000_DiT_P2_384D_6H_6L_EDM_DSM_split2 --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name CIFAR        --dset_start     0 --dset_end   300 --loss_type DSM --exp_name CIFAR_300_DiT_P2_384D_6H_6L_EDM_DSM_split1    --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name CIFAR        --dset_start   300 --dset_end   600 --loss_type DSM --exp_name CIFAR_300_DiT_P2_384D_6H_6L_EDM_DSM_split2    --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name CIFAR        --dset_start     0 --dset_end  1000 --loss_type DSM --exp_name CIFAR_1000_DiT_P2_384D_6H_6L_EDM_DSM_split1   --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name CIFAR        --dset_start  1000 --dset_end  2000 --loss_type DSM --exp_name CIFAR_1000_DiT_P2_384D_6H_6L_EDM_DSM_split2   --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name CIFAR        --dset_start     0 --dset_end  3000 --loss_type DSM --exp_name CIFAR_3000_DiT_P2_384D_6H_6L_EDM_DSM_split1   --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name CIFAR        --dset_start  3000 --dset_end  6000 --loss_type DSM --exp_name CIFAR_3000_DiT_P2_384D_6H_6L_EDM_DSM_split2   --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name CIFAR        --dset_start     0 --dset_end 10000 --loss_type DSM --exp_name CIFAR_10000_DiT_P2_384D_6H_6L_EDM_DSM_split1  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name CIFAR        --dset_start 10000 --dset_end 20000 --loss_type DSM --exp_name CIFAR_10000_DiT_P2_384D_6H_6L_EDM_DSM_split2  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name CIFAR        --dset_start     0 --dset_end 30000 --loss_type DSM --exp_name CIFAR_30000_DiT_P2_384D_6H_6L_EDM_DSM_split1  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
--dataset_name CIFAR        --dset_start 30000 --dset_end 60000 --loss_type DSM --exp_name CIFAR_30000_DiT_P2_384D_6H_6L_EDM_DSM_split2  --patch_size 2 --hidden_size 384 --depth 6 --num_heads 6 --mlp_ratio 4 --eval_sample_size 2048 --eval_batch_size 1024 --lr 1e-4 --nsteps 50000 --batch_size 256
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
python experiment/DiT_learn_curve_CLI.py --record_frequency 0 --eval_sampling_steps 35 --eval_fix_noise_seed \
    $param_name