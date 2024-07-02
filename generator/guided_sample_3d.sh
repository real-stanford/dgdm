python generator/train.py --mode='test' --checkpoint_path='ckpts/dynamics_3d.pt' \
    --diffusion_checkpoint_path='ckpts/diffusion_3d.ckpt' --object_dir='' --save_dir='' \
    --classifier_guidance --num_fingers=16 --grid_size=45 --num_pos=5 --fingers_3d --object_max_num_vertices=512 --ctrlpts_dim=42 --ctrlpts_x_dim=7 --ctrlpts_z_dim=3 \
    --num_workers=0 --num_train_timesteps=15 --num_inference_steps=5 --ema_power=0.85 --batch_size=16 --sub_bs=512 --num_cpus=32 --seed=0
