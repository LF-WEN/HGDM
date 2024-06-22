#ckpt="repair_diff_hgat_ae_x.sde=VP_beta=[0.1,10.0]_adj.sde=VE_beta=[0.2,1.0]_ae=Test_ae_hgat_c=0.01_use_centroidDec=False_lr=0.005"
ckpt="score_model"
saved_name=${ckpt}

#predictor='Euler'
predictor='Reverse'
corrector='Langevin'
#corrector='None'
device="cuda:0"
snr_x="0.2"
eps="0.4"
python sample_zinc.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector
device="cuda:1"
snr_x="0.2"
eps="0.5"
#nohup python sample_zinc.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector  > /dev/null 2>&1 &
device="cuda:2"
snr_x="0.3"
eps="0.9"
#nohup python sample_zinc.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector  > /dev/null 2>&1 &
device="cuda:3"
snr_x="0.4"
eps="0.9"
#nohup python sample_zinc.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector  > /dev/null 2>&1 &
#sleep 2s

#nohup python sample_zinc.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector  > /dev/null 2>&1 &
#sleep 2s
#
#
device="cuda:1"
snr_x="0.2"
eps="0.8"
#nohup python sample_zinc.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector  > /dev/null 2>&1 &
#sleep 2s
#nohup python sample_zinc.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector  > /dev/null 2>&1 &
#sleep 2s
#
#nohup python sample_zinc.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector  > /dev/null 2>&1 &
#sleep 2s
#nohup python sample_zinc.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector  > /dev/null 2>&1 &
#sleep 2s
#nohup python sample_zinc.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector  > /dev/null 2>&1 &
#

#ckpt="diff_hgat_ae_predTrueScore_tangent0_x.sde=VP_beta=[0.1,4.0]_adj.sde=VE_beta=[0.2,1.0]_ae=ae_hgat_ponicare_c=0.01_use_centroidDec=True_lr=0.0001"
#snr_x="0.2"
#device="cuda:0"
#
#nohup python sample_zinc.py $device $ckpt $snr_x 0.7  > /dev/null 2>&1 &
#sleep 2s
#
#nohup python sample_zinc.py $device $ckpt $snr_x 0.8  > /dev/null 2>&1 &
#sleep 2s
#device="cuda:1"
#
#nohup python sample_zinc.py $device $ckpt $snr_x 0.9  > /dev/null 2>&1 &
#sleep 2s
#
#nohup python sample_zinc.py $device $ckpt $snr_x 1.0  > /dev/null 2>&1 &
#sleep 2s
#snr_x="0.5"
#device="cuda:1"
#
#nohup python sample_zinc.py $device $ckpt $snr_x 0.4  > /dev/null 2>&1 &
#sleep 2s
#
#nohup python sample_zinc.py $device $ckpt $snr_x 0.6  > /dev/null 2>&1 &
#sleep 2s