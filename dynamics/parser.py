import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--use_sub_batch', action='store_true', help='use sub batch to avoid OOM')
    parser.add_argument('--sub_bs', type=int, default=1024, help='sub batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs for training')
    parser.add_argument('--num_fingers', type=int, default=1000, help='number of fingers')
    parser.add_argument('--ctrlpts_dim', type=int, default=14)
    parser.add_argument('--ctrlpts_x_dim', type=int, default=7)
    parser.add_argument('--ctrlpts_z_dim', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for optimizer')
    parser.add_argument('--lr_warmup_steps', type=int, default=100, help='learning rate warmup steps for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=500, help='patience for early stopping when training dynamics model')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path to load dynamics model checkpoints')
    parser.add_argument('--save_dir', type=str, help='path to save model checkpoints')
    parser.add_argument('--wandb_id', type=str, default=None, help='wandb id')
    parser.add_argument('--data_dir', type=str, default='', help='path to data directory')
    parser.add_argument('--test_data_dir', type=str, default='', help='path to test data directory')  
    parser.add_argument('--object_dir', type=str, default='', help='path to object directory')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--grid_size', type=int, default=360)
    parser.add_argument('--num_pos', type=int, default=9)
    parser.add_argument('--save_ckpt_step', type=int, default=10, help='step to save model checkpoints')
    parser.add_argument('--val_step', type=int, default=100, help='step to validate model')
    parser.add_argument('--num_train_timesteps', type=int, default=1000, help='number of training timesteps for diffusion model')
    parser.add_argument('--num_timesteps_per_batch', type=int, default=1, help='number of timesteps per batch')
    parser.add_argument('--num_inference_steps', type=int, default=100, help='number of inference steps for diffusion model')
    parser.add_argument('--ema_power', type=float, default=0.75, help='ema power')
    parser.add_argument('--object_max_num_vertices', type=int, default=10, help='max number of vertices for object encoder')
    parser.add_argument('--diffusion_checkpoint_path', type=str, default=None, help='path to load diffusion model checkpoints')
    parser.add_argument('--classifier_guidance', action='store_true', help='use classifier guidance')
    parser.add_argument('--num_cpus', type=int, default=4, help='number of cpus used in parallel for simulation')
    parser.add_argument('--fingers_3d', action='store_true', help='use 3d fingers')
    parser.add_argument('--render_video', action='store_true', help='render video')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()  
    return args