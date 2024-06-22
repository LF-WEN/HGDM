#ckpt="diff_hgat_c=0.01_x.sde=VP_beta=[0.1,5.0]_adj.sde=VE_beta=[0.2,0.8]_ae=Test_ae_hgat_c=0.01_use_centroidDec=False_learnable_c=False_lr=0.01"
ckpt="score_model"
saved_name=${ckpt}
config="default_grid_sample"
#predictor='Euler'
predictor='Reverse'
corrector='Langevin'
#corrector='None'

device="cuda:0"
snr_x="0.25"
eps="0.9"
python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector

#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > grid_sample.log &
device="cuda:1"
snr_x="0.25"
eps="1.0"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > /dev/null 2>&1 &
device="cuda:2"
snr_x="0.3"
eps="1.0"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > /dev/null 2>&1 &
device="cuda:3"
snr_x="0.3"
eps="0.8"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > /dev/null 2>&1 &

#predictor='Euler'
predictor='Reverse'
corrector='Langevin'
#corrector='None'

#ckpt="diff_hgat_c=0.01_x.sde=VE_beta=[0.1,5.0]_adj.sde=VE_beta=[0.2,0.8]_ae=Test_ae_hgat_kl_reg=1e-05_c=0.01_use_centroidDec=True_learnable_c=False_lr=0.001"
#saved_name=${ckpt}
#device="cuda:0"
#snr_x="0.3"
#eps="0.8"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > /dev/null 2>&1 &
#device="cuda:1"
#snr_x="0.3"
#eps="0.4"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > /dev/null 2>&1 &



#device="cuda:2"
#snr_x="0.15"
#eps="0.4"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > /dev/null 2>&1 &
#device="cuda:3"
#snr_x="0.15"
#eps="0.2"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > /dev/null 2>&1 &



