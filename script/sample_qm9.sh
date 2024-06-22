ckpt="score_model"
saved_name=${ckpt}

predictor='Euler'
#predictor='Reverse'
corrector='Langevin'
device="cuda:0"
snr_x="0.15"
eps="0.8"
python sample_qm9.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector

#nohup python sample_qm9.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_qm9_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:1"
snr_x="0.25"
eps="0.9"
#nohup python sample_qm9.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:2"
snr_x="0.1"
eps="0.9"
#nohup python sample_qm9.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:3"
snr_x="0.2"
eps="0.8"
#nohup python sample_qm9.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:0"
snr_x="0.6"
eps="1.0"
#nohup python sample_qm9.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:1"
snr_x="0.6"
eps="0.8"
#nohup python sample_qm9.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:2"
snr_x="0.7"
eps="0.8"
#nohup python sample_qm9.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:3"
snr_x="0.8"
eps="0.8"
#nohup python sample_qm9.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:0"
snr_x="0.9"
eps="0.8"
#nohup python sample_qm9.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:1"
snr_x="0.3"
eps="0.8"
#nohup python sample_qm9.py $device $ckpt $snr_x $eps $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_{$eps}.log &

#nohup python sample_qm9.py $device $ckpt $snr_x 0.7 $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_0.7.log &
#sleep 1s
#nohup python sample_qm9.py $device $ckpt $snr_x 0.8 $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_0.8.log &
#sleep 1s
#nohup python sample_qm9.py $device $ckpt $snr_x 0.9 $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_0.9.log &
#sleep 1s
#nohup python sample_qm9.py $device $ckpt $snr_x 1.0 $saved_name $predictor $corrector >  sample_qm9_{$predictor}_{$snr_x}_1.0.log &
#python sample_qm9.py $device $ckpt 1.0 0.9
