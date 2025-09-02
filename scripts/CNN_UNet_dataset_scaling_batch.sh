#!/bin/bash
#SBATCH -t 12:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=75G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 1-10
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o CNN_UNet_dataset_scaling_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e CNN_UNet_dataset_scaling_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--dataset_name ffhq-32x32  --dset_start 0     --dset_end 300    --loss_type DSM --exp_name FFHQ32_300_UNet_CNN_EDM_DSM_split1    --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name ffhq-32x32  --dset_start 300   --dset_end 600    --loss_type DSM --exp_name FFHQ32_300_UNet_CNN_EDM_DSM_split2    --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name ffhq-32x32  --dset_start 0     --dset_end 1000   --loss_type DSM --exp_name FFHQ32_1000_UNet_CNN_EDM_DSM_split1   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name ffhq-32x32  --dset_start 1000  --dset_end 2000   --loss_type DSM --exp_name FFHQ32_1000_UNet_CNN_EDM_DSM_split2   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name ffhq-32x32  --dset_start 0     --dset_end 3000   --loss_type DSM --exp_name FFHQ32_3000_UNet_CNN_EDM_DSM_split1   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name ffhq-32x32  --dset_start 3000  --dset_end 6000   --loss_type DSM --exp_name FFHQ32_3000_UNet_CNN_EDM_DSM_split2   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name ffhq-32x32  --dset_start 0     --dset_end 10000  --loss_type DSM --exp_name FFHQ32_10000_UNet_CNN_EDM_DSM_split1  --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name ffhq-32x32  --dset_start 10000 --dset_end 20000  --loss_type DSM --exp_name FFHQ32_10000_UNet_CNN_EDM_DSM_split2  --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name ffhq-32x32  --dset_start 0     --dset_end 30000  --loss_type DSM --exp_name FFHQ32_30000_UNet_CNN_EDM_DSM_split1  --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name ffhq-32x32  --dset_start 30000 --dset_end 60000  --loss_type DSM --exp_name FFHQ32_30000_UNet_CNN_EDM_DSM_split2  --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name afhq-32x32  --dset_start 0     --dset_end 300    --loss_type DSM --exp_name AFHQ32_300_UNet_CNN_EDM_DSM_split1    --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name afhq-32x32  --dset_start 300   --dset_end 600    --loss_type DSM --exp_name AFHQ32_300_UNet_CNN_EDM_DSM_split2    --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name afhq-32x32  --dset_start 0     --dset_end 1000   --loss_type DSM --exp_name AFHQ32_1000_UNet_CNN_EDM_DSM_split1   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name afhq-32x32  --dset_start 1000  --dset_end 2000   --loss_type DSM --exp_name AFHQ32_1000_UNet_CNN_EDM_DSM_split2   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name afhq-32x32  --dset_start 0     --dset_end 3000   --loss_type DSM --exp_name AFHQ32_3000_UNet_CNN_EDM_DSM_split1   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name afhq-32x32  --dset_start 3000  --dset_end 6000   --loss_type DSM --exp_name AFHQ32_3000_UNet_CNN_EDM_DSM_split2   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name afhq-32x32  --dset_start 0     --dset_end 10000  --loss_type DSM --exp_name AFHQ32_10000_UNet_CNN_EDM_DSM_split1  --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name afhq-32x32  --dset_start 10000 --dset_end 20000  --loss_type DSM --exp_name AFHQ32_10000_UNet_CNN_EDM_DSM_split2  --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name afhq-32x32  --dset_start 0     --dset_end 30000  --loss_type DSM --exp_name AFHQ32_30000_UNet_CNN_EDM_DSM_split1  --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name afhq-32x32  --dset_start 30000 --dset_end 60000  --loss_type DSM --exp_name AFHQ32_30000_UNet_CNN_EDM_DSM_split2  --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name CIFAR       --dset_start 0     --dset_end 300    --loss_type DSM --exp_name CIFAR_300_UNet_CNN_EDM_DSM_split1     --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name CIFAR       --dset_start 300   --dset_end 600    --loss_type DSM --exp_name CIFAR_300_UNet_CNN_EDM_DSM_split2     --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name CIFAR       --dset_start 0     --dset_end 1000   --loss_type DSM --exp_name CIFAR_1000_UNet_CNN_EDM_DSM_split1    --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name CIFAR       --dset_start 1000  --dset_end 2000   --loss_type DSM --exp_name CIFAR_1000_UNet_CNN_EDM_DSM_split2    --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name CIFAR       --dset_start 0     --dset_end 3000   --loss_type DSM --exp_name CIFAR_3000_UNet_CNN_EDM_DSM_split1    --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name CIFAR       --dset_start 3000  --dset_end 6000   --loss_type DSM --exp_name CIFAR_3000_UNet_CNN_EDM_DSM_split2    --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name CIFAR       --dset_start 0     --dset_end 10000  --loss_type DSM --exp_name CIFAR_10000_UNet_CNN_EDM_DSM_split1   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name CIFAR       --dset_start 10000 --dset_end 20000  --loss_type DSM --exp_name CIFAR_10000_UNet_CNN_EDM_DSM_split2   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name CIFAR       --dset_start 0     --dset_end 30000  --loss_type DSM --exp_name CIFAR_30000_UNet_CNN_EDM_DSM_split1   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
--dataset_name CIFAR       --dset_start 30000 --dset_end 60000  --loss_type DSM --exp_name CIFAR_30000_UNet_CNN_EDM_DSM_split2   --decoder_init_attn True --eval_sample_size 1000 --eval_batch_size 512 --lr 1e-4 --nsteps 50000 --batch_size 256 --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8
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
python experiment/CNN_unet_learn_curve_CLI.py --record_frequency 0 --eval_fix_noise_seed \
    $param_name