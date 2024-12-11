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
from diffusers import UNet3DConditionModel, UNet2DConditionModel, DDPMScheduler,CosineDPMSolverMultistepScheduler,DDIMScheduler
# from transformers import Transformer
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


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
        exponent = torch.arange(self.d_half, dtype=torch.float16)
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
        # d_model=384
        approx_silu = lambda: nn.SiLU()
        self.condition1_proj = Mlp(600*8,600*16,448,act_layer=approx_silu)
        self.condition2_proj = Mlp(50,448*2,448,act_layer=approx_silu)
        self.encodertext=nn.Sequential(
            TransformerEncoder(
                TransformerEncoderLayer(d_model=448, nhead=56),
                num_layers=4
            ),
            Mlp(448,448*2,448,act_layer=approx_silu)
        )
        self.encoder2=nn.Sequential(
            TransformerEncoder(
                TransformerEncoderLayer(d_model=448, nhead=224),
                num_layers=10
            ),
            Mlp(448,448*3,448,act_layer=approx_silu)
        )
        # self.encoder2=Mlp(384,384*6,384)
        self.sin_emb = SinusodialEmbedding(448)
        self.sin_emb2 = SinusodialEmbedding(448)
        self.unet = UNet2DConditionModel(448,in_channels=1,out_channels=1,encoder_hid_dim=448,block_out_channels=(320,640,1280,1280))
        # self.conv1d=nn.Conv1d(448,448,9,padding="same")
        # self.blocks = nn.ModuleList([
        #     DiTBlock(448, 112, mlp_ratio=4.0) for _ in range(200)
        # ])
        # self.final = FinalLayer(448,448)
        self.timesteps = 5
        


        # self.scheduler = CosineDPMSolverMultistepScheduler(num_train_timesteps=self.timesteps)
        
        self.beta_start = 0.2
        # self.beta_end = 0.02
        
        # Define beta schedule
        # Define beta schedule
        self.betas = self.cosine_beta_schedule(self.timesteps).to(torch.float16)

        # Pre-calculate different terms for closed form
        self.alphas = (1.0 - self.betas).to(torch.float16)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(torch.float16)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(torch.float16)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(torch.float16)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(torch.float16)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(torch.float16)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).to(torch.float16)
    def vpsde_beta_t(self,t, T, min_beta, max_beta):
        t_coef = (2 * t - 1) / (T ** 2)
        return 1. - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)
    def vspe_beta_schedule(self, timesteps,min_beta=0.1,max_beta=40):
        schedule_list = np.array([
            self.vpsde_beta_t(t, timesteps, min_beta, max_beta) for t in range(1, timesteps + 1)])
        return torch.from_numpy(schedule_list)
    def cosine_beta_schedule(self, timesteps,s=0.008):
        """
        Creates a cosine schedule for beta values.

        Args:
        - timesteps: The total number of timesteps in the diffusion process.
        - s: Small constant that adjusts the starting point of the schedule.

        Returns:
        - A tensor of beta values for each timestep.
        """
        # Cosine schedule as proposed in the "Improved DDPM" paper
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        schedule_list = np.clip(betas, a_min=0, a_max=0.999)
        return torch.from_numpy(schedule_list)
    
    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1,t.cpu()-1)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward_diffusion(self, x_0, t,mask, device="cuda"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it.
        """
        noise = torch.randn_like(x_0).to(device="cuda")
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        x_noisy = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
                  + sqrt_one_minus_alphas_cumprod_t.to(device) * (noise.to(device)*mask)
        return x_noisy.to(torch.float16), noise.to(torch.float16)
    
    def reverse_diffusion(self, x_noisy, t, noise, mask, device="cuda"):
        """ 
        Reconstruct the original image from the noisy input.
        """
        # Get values at timestep t
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x_noisy.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x_noisy.shape)

        # Predict the original clean data x_0 from noisy input at time t
        x_pred = sqrt_recip_alphas_t.to(device) * (x_noisy.to(device) - sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device))
        x_pred=x_pred*mask
        # Clip the values of x_pred (e.g., to keep it within a valid range like [-1, 1])
        x_pred = torch.clamp(x_pred, -1, 1)

        # Add some noise to simulate the reverse stochastic process (except at t=0)
        if t > 0:
            noise = torch.randn_like(x_noisy).to(device)
            x_pred = x_pred + torch.sqrt(posterior_variance_t) * noise * mask

        return x_pred.to(torch.float16)


    
    

    # def forward_diffusion(self, x_0, t, device="cuda"):
    #     """ 
    #     Takes an image and a timestep as input and 
    #     returns the noisy version of it.
    #     """
    #     noise = torch.randn_like(x_0).to(device="cuda")
    #     x_t = self.scheduler.add_noise(x_0, noise, t)
    #     # print(self.scheduler.alphas_cumprod)
    #     return x_t, noise
    
    # def reverse_diffusion(self, x_noisy, t, noise, device="cuda"):
    #     """ 
    #     Reconstruct the original image from the noisy input.
    #     """
    #     x_t = self.scheduler.step(noise, 20, x_noisy)
    #     return x_t

        

        

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
    
    def normalize_input(self,input_tensor, max_value=1024.0):
        # Normalize to range [-1, 1]
        return ((input_tensor - 512.0) / 512.0).to(torch.float16)
    
    def denormalize_input(self,input_tensor, max_value=1024.0):
        # Normalize to range [-1, 1]
        return (input_tensor*max_value/2+max_value/2).to(torch.int64)



    def forward(self, text_list, proms_list, resps_list=None,spkr_name=None):
        batch_size = len(text_list)  # Assuming all lists have the same length

        # Pad or truncate `resps_list` to shape [batch_size, 384]
        resps_list_padded = []
        for i in range(batch_size):
            if resps_list[i].shape[0] < 448:
                resps_list_padded.append(F.pad(resps_list[i], (0, 448 - resps_list[i].shape[0])))
            else:
                resps_list_padded.append(resps_list[i][:448])
        mask= (resps_list_padded[0]!=0)
        resps_list = torch.stack([self.normalize_input(resps) for resps in resps_list_padded])

        # Pad or truncate `text_list` to shape [batch_size, 50]
        text_list_padded = []
        for i in range(batch_size):
            if text_list[i].shape[0] < 50:
                text_list_padded.append(F.pad(text_list[i], (0, 50 - text_list[i].shape[0])))
            else:
                text_list_padded.append(text_list[i][:50])
        text_list = torch.stack(text_list_padded)

        proms_list_padded = []
        for i in range(batch_size):
            # Get the shape of the current item in proms_list
            prom_shape = proms_list[i].shape[0]

            # If the shape is less than 200, pad it to 200
            if prom_shape < 600:
                # Pad with zeros to make it (200, feature_size), where feature_size is the second dimension
                proms_list_padded.append(F.pad(proms_list[i], (0, 0, 0, 600 - prom_shape)))
            else:
                # If the shape is larger or equal to 200, truncate it to (200, feature_size)
                proms_list_padded.append(proms_list[i][:600, :])

        # Stack the padded tensors back into a tensor
        proms_list = torch.stack(proms_list_padded)
        # Handle conditioning projections for proms_list and text_list
        cond1 = torch.stack([self.normalize_input(proms[:600, :].to(torch.float16).view(-1)) for proms in proms_list])
        cond1 = self.condition1_proj(cond1).unsqueeze(1)

        cond2 = self.condition2_proj(text_list.to(torch.float16)).unsqueeze(1)

        
        cond2 = self.sin_emb.add_pe(cond2)
        cond2 = self.encodertext(cond2)


        cond1=self.sin_emb.add_pe(cond1)
        cond1 = self.encoder2(cond1)
        cond = torch.cat([cond1, cond2], dim=1) # Ensure proper concatenation across batch
        # Initialize loss
        self.loss = 0
        print(mask.sum().item())

        # Loop over time steps and process each batch sample
        for t in range(1, self.timesteps+1):
            t_tensor = torch.tensor([t], dtype=torch.long, device="cuda").expand(batch_size)  # Make time tensor batch-compatible
            x_noisy, noise = self.forward_diffusion(resps_list, t_tensor,mask)
            x_noisy = self.sin_emb2.add_pe(x_noisy)[0]
            x_noisy = x_noisy.unsqueeze(1)
            # x_noisy = torch.cat([x_noisy,cond],dim=1)
            # x_noisy = self.encoder2(x_noisy)
            
            x=x_noisy

            # for block in self.blocks:
            #         x = block(x) 
            # x=x.permute(0,2,1)
            # x=self.conv1d(x)
            # x=x.permute(0,2,1)
            x=x.unsqueeze(1)
            x= self.unet(x,t_tensor,encoder_hidden_states=cond,return_dict=True).sample
            # x=x.squeeze(0)
            
            # x= self.final(x)

            # Calculate loss for each batch and accumulate
            x= x[:,0,:]*mask
            mse_loss = F.mse_loss(x, noise.unsqueeze(1)*mask)
            self.loss += mse_loss

        
        return x

    def generate_audio(self, text_list, proms_list, resps_list=None):
        with torch.no_grad():
            batch_size = len(text_list)  # Assuming all lists have the same length
            resps_list = [torch.randint(low=1,high=1024,size=(200,), dtype=torch.long,device="cuda")]
            # Pad or truncate `resps_list` to shape [batch_size, 384]
            resps_list_padded = []
            for i in range(batch_size):
                if resps_list[i].shape[0] < 448:
                    resps_list_padded.append(F.pad(resps_list[i], (0, 448 - resps_list[i].shape[0])))
                else:
                    resps_list_padded.append(resps_list[i][:448])
            mask= (resps_list_padded[0]!=0)
            resps_list = torch.stack([self.normalize_input(resps) for resps in resps_list_padded])

            # Pad or truncate `text_list` to shape [batch_size, 50]
            text_list_padded = []
            for i in range(batch_size):
                if text_list[i].shape[0] < 50:
                    text_list_padded.append(F.pad(text_list[i], (0, 50 - text_list[i].shape[0])))
                else:
                    text_list_padded.append(text_list[i][:50])
            text_list = torch.stack(text_list_padded)

            proms_list_padded = []
            for i in range(batch_size):
                # Get the shape of the current item in proms_list
                prom_shape = proms_list[i].shape[0]

                # If the shape is less than 200, pad it to 200
                if prom_shape < 600:
                    # Pad with zeros to make it (200, feature_size), where feature_size is the second dimension
                    proms_list_padded.append(F.pad(proms_list[i], (0, 0, 0, 600 - prom_shape)))
                else:
                    # If the shape is larger or equal to 200, truncate it to (200, feature_size)
                    proms_list_padded.append(proms_list[i][:600, :])

            # Stack the padded tensors back into a tensor
            proms_list = torch.stack(proms_list_padded)
            cond1 = torch.stack([self.normalize_input(proms[:600, :].to(torch.float16).view(-1)) for proms in proms_list])
                # Handle conditioning projections for proms_list and text_list
            cond1 = self.condition1_proj(cond1).unsqueeze(1)

            cond2 = self.condition2_proj(text_list.to(torch.float16)).unsqueeze(1)

            
            cond2 = self.sin_emb.add_pe(cond2)
            cond2 = self.encodertext(cond2)


            cond1=self.sin_emb.add_pe(cond1)
            cond1 = self.encoder2(cond1)
            cond = torch.cat([cond1, cond2], dim=1)

            x_noise = resps_list
            # Loop over time steps and process each batch sample
            for t in range(self.timesteps,0,-1):
                t_tensor = torch.tensor([t], dtype=torch.long, device="cuda").expand(batch_size)  # Make time tensor batch-compatible
                x_noisy = self.sin_emb2.add_pe(x_noise)[0]
                # x_noisy = torch.cat(x_noisy,cond,dim=0)
                # x_noisy = self.encoder2(x_noisy)
                x_noisy = x_noisy.unsqueeze(1)
                x=x_noisy

                # for block in self.blocks:
                #         x = block(x) 
                x=x.unsqueeze(1)
                
                x= self.unet(x,t_tensor,encoder_hidden_states=cond,return_dict=True).sample

                x_noise= self.reverse_diffusion(x_noise*mask,t_tensor,x*mask,mask)[0]
            ret =self.denormalize_input(x_noise).squeeze()
            return (ret)

    # def rescale_tensor_to_0_1024(self,tensor, new_min=0, new_max=1024):
    #     # Get the current min and max of the tensor
    #     current_min = tensor.min()
    #     current_max = tensor.max()

    #     # Rescale the tensor to the range [0, 1024]
    #     rescaled_tensor = (tensor - current_min) / (current_max - current_min) * (new_max - new_min) + new_min

    #     return rescaled_tensor          
       
        
    
