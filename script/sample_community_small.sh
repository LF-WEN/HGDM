ckpt="score_model"
saved_name=${ckpt}
config="default_community_small_sample"
#predictor='Euler'
predictor='Reverse'
corrector='Langevin'
#corrector='None'

device="cuda:0"
snr_x="0.05"
eps="0.9"
python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector

#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_cs.log &
#device="cuda:1"
#snr_x="0.25"
#eps="0.8"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > /dev/null 2>&1 &
#device="cuda:2"
#snr_x="0.25"
#eps="0.7"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > /dev/null 2>&1 &
#device="cuda:3"
#snr_x="0.25"
#eps="0.6"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > /dev/null 2>&1 &