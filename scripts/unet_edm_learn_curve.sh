holygpu8a17603

module load python
mamba deactivate
mamba activate torch2
which python

cd ~/Github/DiffusionLearningCurve
python experiment/unet_learn_curve_CLI.py --exp_name MNIST_UNet_CNN_EDM_4blocks_noattn \
    --batch_size 2048 --nsteps 10000 --layers_per_block 1 --model_channels 16 --channel_mult 1 2 3 4 \
    --decoder_init_attn False --record_frequency 0  --eval_sample_size 5000


cd ~/Github/DiffusionLearningCurve
python experiment/unet_learn_curve_CLI.py --exp_name MNIST_UNet_CNN_EDM_4blocks_noattn_denser \
    --batch_size 2048 --nsteps 10000 --layers_per_block 1 --model_channels 16 --channel_mult 1 2 3 4 \
    --decoder_init_attn False --record_frequency 0  --eval_sample_size 5000

cd ~/Github/DiffusionLearningCurve
python experiment/unet_learn_curve_CLI.py --exp_name MNIST_UNet_CNN_EDM_1block_wide128_noattn \
    --batch_size 2048 --nsteps 10000 --layers_per_block 1 --model_channels 128 --channel_mult 1 \
    --decoder_init_attn False --record_frequency 0  --eval_sample_size 5000

cd ~/Github/DiffusionLearningCurve
python experiment/unet_learn_curve_CLI.py --exp_name MNIST_UNet_CNN_EDM_1block_noattn \
    --batch_size 2048 --nsteps 10000 --layers_per_block 1 --model_channels 16 --channel_mult 1 \
    --decoder_init_attn False --record_frequency 0  --eval_sample_size 5000

cd ~/Github/DiffusionLearningCurve
python experiment/unet_learn_curve_CLI.py --exp_name MNIST_UNet_CNN_EDM_2blocks_noattn \
    --batch_size 2048 --nsteps 10000 --layers_per_block 1 --model_channels 16 --channel_mult 1 2 \
    --decoder_init_attn False --record_frequency 0  --eval_sample_size 5000

python experiment/unet_learn_curve_CLI.py --exp_name MNIST_UNet_CNN_EDM_4blocks_wide64_noattn \
    --batch_size 2048 --nsteps 10000 --layers_per_block 1 --model_channels 64 --channel_mult 1 2 3 4 \
    --decoder_init_attn False --record_frequency 0  --eval_sample_size 5000


cd ~/Github/DiffusionLearningCurve
python experiment/unet_FFHQ_learn_curve_CLI.py --exp_name FFHQ_UNet_CNN_EDM_4blocks_wide64_attn_pilot2 \
    --decoder_init_attn True --record_frequency 0  --eval_sample_size 64 --eval_batch_size 512 \
    --lr 2e-4 --nsteps 10000 --batch_size 256 \
    --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 -R 0 10000 50000
    

cd ~/Github/DiffusionLearningCurve
python experiment/unet_FFHQ_learn_curve_CLI.py --exp_name FFHQ_UNet_CNN_EDM_4blocks_wide64_attn_pilot3 \
    --decoder_init_attn True --record_frequency 0  --eval_sample_size 1000 --eval_batch_size 512 \
    --lr 2e-4 --nsteps 50000 --batch_size 256 \
    --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
    
cd ~/Github/DiffusionLearningCurve
python experiment/unet_FFHQ_learn_curve_CLI.py --exp_name FFHQ_UNet_CNN_EDM_4blocks_wide64_attn_pilot_fixednorm \
    --decoder_init_attn True --record_frequency 0  --eval_sample_size 1000 --eval_batch_size 512 \
    --lr 2e-4 --nsteps 50000 --batch_size 256 \
    --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 

cd ~/Github/DiffusionLearningCurve
python experiment/unet_FFHQ_learn_curve_CLI.py --dataset_name AFHQ --exp_name AFHQ_UNet_CNN_EDM_4blocks_wide64_attn_pilot_fixednorm \
    --decoder_init_attn True --record_frequency 0  --eval_sample_size 1000 --eval_batch_size 512 \
    --lr 2e-4 --nsteps 50000 --batch_size 256 \
    --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 --layers_per_block 1
# --cond=0 --arch=ddpmpp --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15



cd ~/Github/DiffusionLearningCurve
python experiment/unet_FFHQ_learn_curve_CLI.py --dataset_name CIFAR --exp_name CIFAR10_UNet_CNN_EDM_3blocks_wide128_attn_pilot_fixednorm \
    --decoder_init_attn True --record_frequency 0  --eval_sample_size 2000 --eval_batch_size 1024 \
    --lr 0.001 --nsteps 50000 --batch_size 512 \
    --model_channels 128 --channel_mult 2 2 2 --attn_resolutions 16 --eval_sampling_steps 20 --layers_per_block 1

# torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp


# layers_per_block = 1
# decoder_init_attn = True
# attn_resolutions = [8, 16,]
# model_channels = 64
# channel_mult = [1, 2, 4, 4]
# batch_size = 512
# nsteps = 10000
# lr = 1e-4
# record_frequency = 100
# exp_name = "ffhq64_edm_model_pilot"
# eval_sample_size = 5000
# eval_batch_size = 1024 # Process in batches of 1000

# ranges = [(0, 10, 1), (10, 50, 2), (50, 100, 4), (100, 500, 8), (500, 2500, 16), (2500, 5000, 32), (5000, 10000, 64)]
# record_times = generate_record_times(ranges)