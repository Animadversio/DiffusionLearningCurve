holygpu8a17603

module load python
mamba deactivate
mamba activate torch2
export PATH="$HOME/.conda/envs/torch2/bin:${PATH}"
which python

cd ~/Github/DiffusionLearningCurve
python experiment/CNN_unet_learn_curve_CLI.py --exp_name words32x32_50k_UNet_CNN_EDM_4blocks_noattn \
    --dataset_name words32x32_50k \
    --batch_size 2048 --nsteps 50000 --layers_per_block 1 --model_channels 32 --channel_mult 1 2 3 4 \
    --decoder_init_attn False --record_frequency 0  --eval_sample_size 5000 --lr 1e-4 --eval_sample_size 2048 --eval_batch_size 2048 --eval_sampling_steps 40

 
cd ~/Github/DiffusionLearningCurve
python experiment/CNN_unet_learn_curve_CLI.py --exp_name FFHQ_fix_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm \
    --dataset_name FFHQ_fix_words \
    --decoder_init_attn True --record_frequency 0  --eval_sample_size 1000 --eval_batch_size 512 \
    --lr 1e-4 --nsteps 50000 --batch_size 256 \
    --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 

cd ~/Github/DiffusionLearningCurve
python experiment/CNN_unet_learn_curve_CLI.py --exp_name FFHQ_random_words_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm \
    --dataset_name FFHQ_random_words_jitter \
    --decoder_init_attn True --record_frequency 0  --eval_sample_size 1000 --eval_batch_size 512 \
    --lr 1e-4 --nsteps 50000 --batch_size 256 \
    --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 

cd ~/Github/DiffusionLearningCurve
python experiment/CNN_unet_learn_curve_CLI.py --exp_name FFHQ_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm \
    --dataset_name FFHQ \
    --decoder_init_attn True --record_frequency 0  --eval_sample_size 1000 --eval_batch_size 512 \
    --lr 1e-4 --nsteps 50000 --batch_size 256 \
    --model_channels 128 --channel_mult 1 2 2 2 --attn_resolutions 8 
