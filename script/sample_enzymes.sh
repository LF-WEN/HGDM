#ckpt="repair_newae_lr_schedule_diff_Atimescale_c=0.01_x.sde=VP_beta=[0.1,1.0]_adj.sde=VE_beta=[0.2,1.0]"
#saved_name="repair_newae_lr_schedule_diff_Atimescale_c=0.01_x.sde=VP_beta=[0.1,1.0]_adj.sde=VE_beta=[0.2,1.0]_4500"
#ckpt="repair_diff_ae_c=0.01_x.sde=VP_beta=[0.1,1.0]_adj.sde=VE_beta=[0.2,1.0]_ae=Test_ae_hgat_kl=1e-05_c=0.01_use-centroidDec=True_learnable-c=False_lr=0.01"
ckpt="score_model"
saved_name=${ckpt}
config="default_enzymes_sample"
#predictor='Euler'
predictor='Reverse'
corrector='Langevin'
#corrector='None'

device="cuda:0"
snr_x="0.5"
eps="0.9"
python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector

#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_enzymes_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:1"
snr_x="0.7"
eps="0.4"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_enzymes_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:0"
#snr_x="0.1"
eps="0.7"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_enzymes_{$predictor}_{$snr_x}_{$eps}.log &
device="cuda:1"
#snr_x="0.1"
eps="0.9"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_enzymes_{$predictor}_{$snr_x}_{$eps}.log &



#ckpt="repair_diff_ae_c=0.01_x.sde=VP_beta=[0.1,2.0]_adj.sde=VE_beta=[0.2,1.0]_ae=Test_ae_hgat_c=0.01_use-centroidDec=True_learnable-c=False_lr=0.01"
#saved_name="repair_diff_ae_c=0.01_x.sde=VP_beta=[0.1,2.0]_adj.sde=VE_beta=[0.2,1.0]_ae=Test_ae_hgat_c=0.01_use-centroidDec=True_learnable-c=False_lr=0.01"
#device="cuda:0"
#snr_x="0.3"
#eps="0.1"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_enzymes_{$predictor}_{$snr_x}_{$eps}.log &
#device="cuda:1"
##snr_x="0.1"
#eps="0.8"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_enzymes_{$predictor}_{$snr_x}_{$eps}.log &
#device="cuda:2"
##snr_x="0.1"
#eps="0.6"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_enzymes_{$predictor}_{$snr_x}_{$eps}.log &
#device="cuda:3"
##snr_x="0.1"
#eps="0.4"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_enzymes_{$predictor}_{$snr_x}_{$eps}.log &
#device="cuda:4"
##snr_x="0.1"
#eps="0.3"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_enzymes_{$predictor}_{$snr_x}_{$eps}.log &
#device="cuda:5"
##snr_x="0.1"
#eps="0.2"
#nohup python sample_synthetic.py $config $device $ckpt $snr_x $eps $saved_name $predictor $corrector > sample_enzymes_{$predictor}_{$snr_x}_{$eps}.log &




