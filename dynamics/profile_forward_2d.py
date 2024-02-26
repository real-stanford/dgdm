import math
import torch
import torch.nn as nn

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:  # original raw input "x" is also included in the output
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(input_dims, multires, i=0, scalar_factor=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x/scalar_factor)
    return embed, embedder_obj.out_dim

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ProfileForward2DModel(nn.Module):
    def __init__(self, W=256, params_ch=400, ori_ch=1, pos_ch=2, output_ch=3, object_ch=20):
        super(ProfileForward2DModel, self).__init__()
        self.W = W
        self.params_ch = params_ch
        self.ori_ch = ori_ch
        self.pos_ch = pos_ch
        self.output_ch = output_ch
        self.ori_embed, ori_embed_dim = get_embedder(ori_ch, 4, 0, scalar_factor=1)
        self.pos_embed, pos_embed_dim = get_embedder(pos_ch, 4, 0, scalar_factor=1)
        self.ori_ch = ori_embed_dim
        self.pos_ch = pos_embed_dim
        self.pose_embed_dim = ori_embed_dim + pos_embed_dim
        self.time_embed_dim = W
        self.time_encoder = nn.Sequential(
            nn.Linear(W // 2, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.object_encode_dim = W
        self.object_encoder = nn.Sequential(
            nn.Linear(object_ch, self.object_encode_dim),
            nn.ReLU(),
            nn.Linear(self.object_encode_dim, self.object_encode_dim),
        )
        self.gripper_encoder = nn.Sequential(
            nn.Linear(params_ch, W),
            nn.ReLU(),
            nn.Linear(W, W),
        )
        self.gripper_encode_dim = W
        self.linears = nn.Sequential(
            nn.Linear(self.gripper_encode_dim + self.pose_embed_dim + self.time_embed_dim + self.object_encode_dim, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.BatchNorm1d(W),
            nn.ReLU(),
        )
        self.output = nn.Linear(W, output_ch)
        
    def forward(self, x_ctrl, x_ori, x_pos, timesteps, object_vertices):
        '''
        input: 
            ctrlpts [batch_size, 400]
            ori [batch_size, 1]
            pos [batch_size, 2]
            timesteps [batch_size,]
            object_vertices [batch_size, 20]
        output: 
            profile [batch_size, 9]
        '''
        x_ctrl = self.gripper_encoder(x_ctrl)
        x_ori = self.ori_embed(x_ori)
        x_pos = self.pos_embed(x_pos)
        x_pose = torch.cat([x_ori, x_pos], dim=1)
        x_object = self.object_encoder(object_vertices)
        time_emb = self.time_encoder(timestep_embedding(timesteps, self.W // 2))
        x = self.linears(torch.cat([x_object, x_ctrl, x_pose, time_emb], dim=1))
        x = self.output(x)
        return x
