import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from .base import MultiEmbedding,Embedding, Base
from torch import Tensor
from torch.distributions import Categorical
from tqdm import trange
import numpy as np
import math
from diffusers import UNet3DConditionModel, UNet2DConditionModel
# from transformers import Transformer
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

class FiLM(nn.Module):
    def __init__(self, d_model):
        super(FiLM, self).__init__()
        self.linear = nn.Linear(d_model, d_model * 2)  # For scale and shift

    def forward(self, x, conditioning):
        gamma, beta = self.linear(conditioning).chunk(2, dim=-1)
        return gamma * x + beta
    
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x_norm = self.norm1(x)
        
        # Apply multihead self-attention
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        
        # Residual connection
        x = x + attn_output
        
        # Layer normalization before MLP
        x_norm = self.norm2(x)
        
        # Apply MLP
        x = x + self.mlp(x_norm)
        
        return x
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x

def _create_mask(l, device):
    """1 is valid region and 0 is invalid."""
    seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
    stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
    return (seq < stop).float()  # (b t)

def list_to_tensor(x_list, pattern="t b c -> b t c"):
    """
    Args:
        x_list: [(t d)]
    Returns:
        x: (? ? ?)
        m: (? ? ?), same as x
    """
    l = list(map(len, x_list))
    x = rearrange(pad_sequence(x_list), pattern)
    m = _create_mask(l, x_list[0].device)
    m = m.t().unsqueeze(-1)  # (t b 1)
    m = rearrange(m, pattern)
    m = m.to(x)
    return x, m

def _unsqueeze_list(x_list, axis=-1):
    return [x.unsqueeze(dim=axis) for x in x_list]
def _join(x: tuple[Tensor], sep: Tensor):
    """
    Args:
        x: (k t d)
        sep: (d)
    """
    ret = x[0]
    for i in range(1, len(x)):
        ret = torch.cat((ret, x[i]), dim=0)
    return ret
def _samplewise_merge_tensors(*l, sep: Tensor | None):
        if sep is None:
            cat = torch.cat
        else:
            cat = partial(_join, sep=sep)
        return [*map(cat, zip(*l))]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)
    
class FiLM(nn.Module):
    def __init__(self, d_model):
        super(FiLM, self).__init__()
        self.linear = nn.Linear(d_model, d_model * 2)  # For scale and shift

    def forward(self, x, conditioning):
        gamma, beta = self.linear(conditioning).chunk(2, dim=-1)
        return gamma * x + beta
    
class SinusodialEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        exponent = torch.arange(self.d_half, dtype=torch.float32)
        exponent = exponent / self.d_half
        omega = torch.exp(-math.log(1e4) * exponent)
        self.omega: torch.Tensor
        self.register_buffer("omega", omega, persistent=False)

    @property
    def d_half(self):
        assert self.d_model % 2 == 0, "Only support even d_model."
        return self.d_model // 2

    def forward(self, x):
        """
        Args:
            x: (...)
        Returns:
            pe: (... d)
        """
        omega = self.omega

        while omega.dim() <= x.dim():
            omega = omega.unsqueeze(0)  # (... d)

        x = x.unsqueeze(-1)  # (... 1)
        x = omega * x
        x = torch.cat([x.sin(), x.cos()], dim=-1)

        return x

    def get_pe(self, n: int):
        """
        Args:
            n: int
        Returns:
            pe: (n d)
        """
        device = self.omega.device
        return self.forward(torch.arange(n, device=device))

    def add_pe(self, x):
        """
        Args:
            x: (b t c)
        """
        e = self.get_pe(x.shape[0])  # t d
        e = e[None]  # b t d
        x = x + e
        return x

class AR(nn.Module):
    @property
    def n_resp_levels(self):
        return 1

    @property
    def casual(self):
        return True

    @property
    def use_stop_token(self):
        return True

    @property
    def norm_type(self):
        return "ln"

    @property
    def resp_loss_only(self):
        return False
    
    def _prune(self, l: Tensor):
        indices = (l == self.stop_token).nonzero()
        if len(indices) == 0:
            return l
        return l[: indices.min().item()]

    def __init__(self, d_model, n_steps, n_tokens, max_n_levels, n_heads, num_layers):
        super().__init__()
        d_model=64
        self.text_emb = Embedding(n_tokens, d_model)
        self.proms_emb = MultiEmbedding(max_n_levels, n_tokens, d_model)
        self.resps_emb = MultiEmbedding(1, n_tokens, d_model)
        self.sin_emb = SinusodialEmbedding(d_model)
        self.time_emb = Embedding(201, d_model)
        self.encoder1=nn.Sequential(
            TransformerEncoder(
                TransformerEncoderLayer(d_model=64, nhead=16),
                num_layers=4
            ),
            Mlp(64,256,32)
        )
        self.encoder2= nn.Sequential(
            TransformerEncoder(
                TransformerEncoderLayer(d_model=32, nhead=16),
                num_layers=4
            ),
            Mlp(32,128,16)

        )
        self.bottleneck = TransformerEncoder(
                TransformerEncoderLayer(d_model=16, nhead=16),
                num_layers=4
            )
        self.blocks = nn.ModuleList([
            DiTBlock(16, 16, mlp_ratio=3.0) for _ in range(4)
        ])
        self.pre_decout=nn.Sequential(
            nn.Linear(16, 64),  # Upsample (increase embedding dimension)
            nn.GELU())
        # self.pre_decrout2=nn.Sequential(
        #     nn.Linear(32, d_model),  # Upsample (increase embedding dimension)
        #     nn.GELU())
        self.decout=TransformerDecoder(
                TransformerDecoderLayer(d_model=d_model, nhead=16),
                num_layers=5
            )
        self.output=Mlp(d_model,d_model*3,d_model)
        self.num_steps=self.timesteps = 175
        self.beta_start = 0.008
        self.beta_end = 0.02
        
        # Define beta schedule
        # Define beta schedule
        self.betas = self.cosine_beta_schedule(self.timesteps, self.beta_start).to(torch.float32)

        # Pre-calculate different terms for closed form
        self.alphas = (1.0 - self.betas).to(torch.float32)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(torch.float32)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(torch.float32)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(torch.float32)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(torch.float32)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(torch.float32)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).to(torch.float32)

    
    def cosine_beta_schedule(self, timesteps, s=0.0001):
        """
        Creates a cosine schedule for beta values.

        Args:
        - timesteps: The total number of timesteps in the diffusion process.
        - s: Small constant that adjusts the starting point of the schedule.

        Returns:
        - A tensor of beta values for each timestep.
        """
        # Cosine schedule as proposed in the "Improved DDPM" paper
        steps = np.arange(timesteps + 1, dtype=np.float64)
        alphas_cumprod = np.cos(((steps / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        return torch.tensor(betas, dtype=torch.float32)
    
    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1,t.cpu()-1)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward_diffusion(self, x_0, t, device="cuda"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it.
        """
        x_0=x_0[0].float()
        noise = torch.randn_like(x_0).to(device="cuda")
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        x_noisy = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
                  + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        return x_noisy.to(torch.float16), noise.to(torch.float16)
    
    def reverse_diffusion(self, x_noisy, t, noise, device="cuda"):
        """ 
        Reconstruct the original image from the noisy input.
        """
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x_noisy.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
        if(t==self.timesteps):
            return torch.tensor(2.00, dtype=torch.float32).to(device) * (x_noisy.to(device) - sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)).to(torch.float16)
        # Mean reconstruction
        x_pred = sqrt_recip_alphas_t.to(device) * (x_noisy.to(device) - sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device))
        
        return x_pred.to(torch.float16)

        

        

    @property
    def stop_token(self):
        if not self.use_stop_token:
            raise ValueError("Not using stop token!")
        return self.n_tokens
    def float_to_int(self, float_tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts a float tensor back to an int tensor using argmax.
        """
        int_tensor = torch.argmax(float_tensor, dim=-1)
        return int_tensor

    def forward(self, text_list, proms_list, resps_list=None):
        if resps_list[0].shape[0]<356:
            resps_list=[F.pad(resps_list[0], (0,356-resps_list[0].shape[0]))]
        else:
            resps_list=[resps_list[0][:356]]
        if text_list[0].shape[0]<50:
            text_list = [F.pad(text_list[0], (0,50-text_list[0].shape[0]))]
        else:
            text_list=[text_list[0][:50]]
        epsilon = 1e-10
        text_emb = self.text_emb(text_list)
        proms_emb = self.proms_emb([proms_list[0][:100,:]])
        resps_emb = self.resps_emb(_unsqueeze_list(resps_list))
        
        self.loss = 0
        if resps_list is not None:
            for t in range(1, self.num_steps+1 ):
                t_tensor = torch.tensor([t], dtype=torch.long, device="cuda")
                x_noise, noise = self.forward_diffusion(resps_emb, t_tensor)
                t_emb = self.time_emb([t_tensor])

                x_noisy=torch.cat([x_noise,proms_emb[0],t_emb[0],text_emb[0]], dim=0)
                x_noisy = self.sin_emb.add_pe(x_noisy)[0]
                
                x_pred1= self.encoder1(x_noisy)
                x_pred2= self.encoder2(x_pred1)
                x_mem= self.bottleneck(x_pred2)
                for block in self.blocks:
                    x_mem = block(x_mem)  

                x_pred = self.pre_decout(x_mem)
                # x_pred = self.pre_decrout2(x_pred)
                x_noise = self.sin_emb.add_pe(x_noise)[0]
                # x_pred = self.decoder2(x_noise,x_pred)
                x_pred = self.decout(x_noise,x_pred)
                x_pred=self.output(x_pred)
                # x_pred=x_pred[:300,:]
                # cosine_loss = 1-F.cosine_similarity(x_pred,noise).mean()
                mse_loss=F.mse_loss(x_pred, noise)
                # x_pred=F.softmax(x_pred, dim=-1)
                # noise = F.softmax(noise,dim=-1)
                # kl_loss = F.kl_div((x_pred+epsilon).log(), noise, reduction='batchmean')
                
                self.loss += (mse_loss)
            return noise
        


    def generate_audio(self, text_list, proms_list, resps_list=None):
        with torch.no_grad():
            resps_list = [torch.randint(low=0, high=256, size=(300,), dtype=torch.long,device="cuda")]
            text_emb = self.text_emb(text_list)
            proms_emb = self.proms_emb([proms_list[0][:100,:]])
            resps_emb = self.resps_emb(_unsqueeze_list(resps_list))
            x_noise = resps_emb[0].to(torch.float16)
            for t in range(self.timesteps,0,-1):
                t_tensor = torch.tensor([t], dtype=torch.long, device="cuda")
                # x_noisy, noise = self.forward_diffusion(resps_emb, t_tensor)
                t_emb = self.time_emb([t_tensor])

                x_noisy=torch.cat([x_noise,proms_emb[0],t_emb[0],text_emb[0]], dim=0).to(torch.float16)
                x_noisy = self.sin_emb.add_pe(x_noisy)[0]
                x_pred1= self.encoder1(x_noisy)
                x_pred2= self.encoder2(x_pred1)
                x_mem= self.bottleneck(x_pred2)
                for block in self.blocks:
                    x_mem = block(x_mem)  

                x_pred = self.pre_decout(x_mem)
                # x_pred = self.pre_decrout2(x_pred)
                x_noise = self.sin_emb.add_pe(x_noise)[0]
                # x_pred = self.decoder2(x_noise,x_pred)
                x_pred = self.decout(x_noise,x_pred)
                x_pred=self.output(x_pred)
                x_noise = self.reverse_diffusion(x_noise,t_tensor, x_pred) 

            # x_noise=x_noise[:300,:]
            ret = self.find_closest_embedding(x_noise,self.resps_emb)
            return ret
            # return resp
            
        
    # Function to find the closest embedding
    def find_closest_embedding(self,predicted_embeddings, embedding_layer):
        closest_indices = torch.zeros(predicted_embeddings.size(0), dtype=torch.long, device="cuda")
        for i, predicted_embedding in enumerate(predicted_embeddings):
            # Compute the distances between the predicted embedding and all embeddings in the layer
            distances = torch.norm(embedding_layer.weight[0] - predicted_embedding,dim=1)
            # Find the index of the closest embedding
            closest_index = torch.argmin(distances).item()
            # Append the closest index to the list
            closest_indices[i] = closest_index

        # # return closest_indices
        # closest_indices = torch.zeros(predicted_embeddings.size(0), dtype=torch.long, device="cuda")

        # # Normalize the embeddings
        # normalized_pred = torch.nn.functional.normalize(predicted_embeddings, p=2, dim=1)
        # normalized_emb_layer = torch.nn.functional.normalize(embedding_layer.weight, p=2, dim=1)

        # # Compute cosine similarity
        # similarities = torch.matmul(normalized_pred, normalized_emb_layer.T)

        # # Find the index with the maximum similarity (closest vector)
        # closest_indices = torch.argmax(similarities, dim=0)

        return closest_indices
