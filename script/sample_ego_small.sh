#ckpt="Test_repair_diff_hgat_c=0.01_x.sde=VP_beta=[0.1,1.0]_adj.sde=VP_beta=[0.1,1.0]_ae=ae_hgat_c=0.01_use_centroidDec=True_learnable_c=False_lr=0.01"
ckpt="score_model"
saved_name=${ckpt}
config="default_ego_small_sample"

predictor='Euler'
#predictor='Reverse'
corrector='Langevin'
#corrector='None'

device="cuda:0"
snr_x="0.25"
eps="1.0"
python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector

#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_es.log &
device="cuda:1"
snr_x="0.6"
eps="0.8"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_es.log &
device="cuda:1"
snr_x="0.7"
eps="0.9"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > /dev/null 2>&1 &
