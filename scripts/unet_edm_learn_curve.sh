holygpu8a17603

module load python
mamba deactivate
mamba activate torch2
which python

cd ~/Github/DiffusionLearningCurve
python experiment/unet_learn_curve_CLI.py --exp_name MNIST_UNet_CNN_EDM_4blocks_noattn \
    --batch_size 2048 --nsteps 10000 --layers_per_block 1 --model_channels 16 --channel_mult 1 2 3 4 \
    --decoder_init_attn False --record_frequency 0  --eval_sample_size 5000

python experiment/unet_learn_curve_CLI.py --exp_name MNIST_UNet_CNN_EDM_4blocks_wide64_noattn \
    --batch_size 2048 --nsteps 10000 --layers_per_block 1 --model_channels 64 --channel_mult 1 2 3 4 \
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

