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
import torch.nn as nn

import torch
import torch.nn as nn

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning,
    cross-attention for text phonemes, and conditioning on speaker and timestep embeddings.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size

        # Self-Attention Layer
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, **block_kwargs)
        
        # Cross-Attention Layer for Text Phonemes
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, **block_kwargs)


        # Cross-Attention Layer for speaker embeddings
        self.norm22 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn2 = nn.MultiheadAttention(hidden_size, num_heads=num_heads, **block_kwargs)
        
        # MLP Layer
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        self.timestep_fc = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, x, speaker_emb, text_phonemes, timestep_emb,mask):
        mask=mask.unsqueeze(0).unsqueeze(2)
        x=x*mask
        x=x.transpose(0,1)
        # Self-Attention Block
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        
        # Cross-Attention Block with Text Phonemes
        x_norm = self.norm2(x)
        text_phonemes = text_phonemes.transpose(0,1)
        cross_attn_output, _ = self.cross_attn(x_norm, text_phonemes, text_phonemes)

        x_norm = self.norm22(x)
        speaker_emb = speaker_emb.transpose(0,1)
        cross_attn_output2, _ = self.cross_attn(x_norm, speaker_emb, speaker_emb)
        x = x + cross_attn_output+cross_attn_output2

        timestep_scale_shift = self.timestep_fc(timestep_emb)
        scale = timestep_scale_shift[:, :self.hidden_size]
        shift = timestep_scale_shift[:, self.hidden_size:]

        scale = scale.unsqueeze(0)  # Shape: (1, batch_size, hidden_size)
        shift = shift.unsqueeze(0)  

        # Apply Layer Normalization
        x_norm = self.norm3(x)
        
        # Apply FiLM Modulation
        x_modulated = x_norm * (1 + scale) + shift  # FiLM modulation

        # MLP Block
        x = x + self.mlp(x_modulated)

        return x.transpose(0,1)*mask


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
        self.timesteps = 100
        d_model=32
        approx_silu = lambda: nn.SiLU()
        self.text_emb = nn.Embedding(1025,d_model,padding_idx=0)
        self.proms_emb = MultiEmbedding(max_n_levels, 1025, d_model)
        self.resps_emb = nn.Embedding(1025,d_model,padding_idx=0)
        self.time_emb = nn.Embedding(self.timesteps+1, d_model)


        self.encodertext=nn.Sequential(
            TransformerEncoder(
                TransformerEncoderLayer(d_model=d_model, nhead=16),
                num_layers=2
            ),
            Mlp(d_model,d_model*2,d_model,act_layer=approx_silu,drop=0.01)
        )

        self.encoder2=nn.Sequential(
            TransformerEncoder(
                TransformerEncoderLayer(d_model=d_model, nhead=16),
                num_layers=2
            ),
            Mlp(d_model,d_model*3,d_model,act_layer=approx_silu,drop=0.01)
        )
        # self.encoder2=Mlp(384,384*6,384)
        self.sin_emb = SinusodialEmbedding(d_model)
        self.sin_emb2 = SinusodialEmbedding(d_model)
        self.token_emb = nn.Embedding(num_embeddings=1025,embedding_dim=d_model)
        # self.unet = UNet2DConditionModel(d_model,in_channels=1,out_channels=1,encoder_hid_dim=d_model,block_out_channels=(8,8,8,8),only_cross_attention=True,norm_num_groups=8)
        # self.conv1d=nn.Conv1d(448,448,9,padding="same")
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, 16, mlp_ratio=4.0) for _ in range(8)
        ])
        self.final = nn.Linear(d_model,1025)
        # self.encoder3=nn.Sequential(
        #     TransformerEncoder(
        #         TransformerEncoderLayer(d_model=d_model, nhead=16),
        #         num_layers=4
        #     ),
        #     Mlp(d_model,d_model*3,d_model,act_layer=approx_silu,drop=0.01)
        #  )
        # self.decout=TransformerDecoder(
        #         TransformerDecoderLayer(d_model=d_model, nhead=16),
        #         num_layers=5
        #     )
        
        # self.beta_start = 0.2
        # self.beta_end = 0.02
        self.num_classes=1025
        # Define beta schedule
        self.betas = self.cosine_beta_schedule(self.timesteps+1).to(torch.float16)

        # # Pre-calculate different terms for closed form
        # self.alphas = (1.0 - self.betas).to(torch.float16)
        # self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(torch.float16)
        # self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(torch.float16)
        # self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(torch.float16)
        # self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(torch.float16)
        # self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(torch.float16)
        # self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).to(torch.float16)

        self.q_onestep_mats = torch.stack([self._get_absorbing_transition_mat(t)
                               for t in range(0, self.timesteps)],dim=0).to("cuda:0").to(torch.float16)
        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.timesteps):
            q_mat_t = torch.tensordot(q_mat_t, self.q_onestep_mats[t], dims=[[1], [0]])
            q_mats.append(q_mat_t)
        self.q_mats=torch.stack(q_mats,dim=0).to("cuda:0").to(torch.float16)
        self.eps = 1.e-6
        self.transpose_q_onestep_mats=torch.transpose(self.q_onestep_mats, 1, 2).to("cuda:0").to(torch.float16)
        
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
    def linear_beta_schedule(self,timesteps,start,stop):
         return torch.linspace(start, stop,timesteps)
    
    def create_transition_matrix(self, beta_t):
        num_classes = 1025
        mat = torch.full((num_classes, num_classes), beta_t / num_classes).to(torch.float16)
        diag_indices = torch.arange(num_classes)
        mat[diag_indices, diag_indices] = 1.0 - beta_t * (num_classes - 1) / num_classes
        return mat
    
    def _get_absorbing_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Has an absorbing state for pixelvalues self.num_pixel_vals//2.

        Args:
          t: timestep. integer scalar.

        Returns:
          Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
        """
        beta_t = self.betas[t].numpy()

        diag = np.full(shape=(1025,), fill_value=1. - beta_t,
                       dtype=np.float64)
        mat = np.diag(diag, k=0)
        # Add beta_t to the num_pixel_vals/2-th column for the absorbing state.
        mat[:, 1025 // 2] += beta_t

        return torch.from_numpy(mat)

    
    def _at(self, a, t, x):
        B,W= x.shape
        a_t = torch.index_select(a, dim=0, index=t)
        # assert a_t.shape == (x.shape[0], 1025, 1025)
        # out = a_t[x.tolist()]
        x_onehot = F.one_hot(x.view(B, -1).to(torch.int64), num_classes=1025).to(torch.float16)
        out = torch.matmul(x_onehot, a_t)
        out = out.view(B,W,1025)
        return out
    
    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        """Compute logits of q(x_{t-1} | x_t, x_start)."""

        # if x_start_logits:
        #     assert x_start.shape == x_t.shape + (self.num_pixel_vals,), (
        #         x_start.shape, x_t.shape)
        # else:
        #     assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
        if x_start_logits:
            t_1 = torch.where(t == 0, t, t - 1)
            fact2 = self._at_onehot(self.q_mats, t_1,
                                    F.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            t_1 = torch.where(t == 0, t, t-1)
            fact2 = self._at(self.q_mats, t_1, x_start)
            tzero_logits = torch.log(
                F.one_hot(x_start.to(torch.int64), num_classes=self.num_pixel_vals)
                + self.eps)

        # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
        # where x_{-1} == x_start. This should be equal the log of x_0.
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        # t_broadcast = np.expand_dims(t, tuple(range(1, out.ndim)))
        t_broadcast = torch.reshape(t, ([out.shape[0]] + [1] * (len(out.shape) - 1)))
        return torch.where(t_broadcast == 0, tzero_logits,
                           out)
    
    def _at_onehot(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
          a: np.ndarray: plain NumPy float64 array of constants indexed by time.
          t: jnp.ndarray: Jax array of time indices, shape = (bs,).
          x: jnp.ndarray: jax array, shape (bs, ..., num_pixel_vals), float32 type.
            (Noisy) data. Should be of one-hot-type representation.

        Returns:
          out: jnp.ndarray: Jax array. output of dot(x, a[t], axis=[[-1], [1]]).
            shape = (bs, ..., num_pixel_vals)
        """

        # x.shape = (bs, channels, height, width, num_pixel_vals)
        # a[t]shape = (bs, num_pixel_vals, num_pixel_vals)
        # out.shape = (bs, height, width, channels, num_pixel_vals)
        B, W, _ = x.shape
        a_t = torch.index_select(a, dim=0, index=t)
        # assert a_t.shape == (x.shape[0], 1025, 1025)
        x = x.view(B, -1, 1025)
        out = torch.matmul(x, a_t)
        out = out.view(B,W, 1025)
        return out
    def p_sample(self,model_logits,t,x):
        noise = torch.rand(size=x.shape+(1025,)).to(x.device)
        pred_x_start_logits = model_logits

        # t_broadcast = np.expand_dims(t, tuple(range(1, model_logits.ndim)))
        t_broadcast = torch.reshape(t, ([model_logits.shape[0]] + [1] * (len(model_logits.shape) - 1)))
        model_logits = torch.where(t_broadcast == 0,
                                    pred_x_start_logits,
                                       self.q_posterior_logits(pred_x_start_logits, x,
                                                               t, x_start_logits=True)
                                       )
        
        nonzero_mask = (t != 0).to(x.dtype).reshape(x.shape[0],
                                                        *([1] * (len(x.shape))))
        # For numerical precision clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))

        sample = torch.argmax(model_logits + nonzero_mask * gumbel_noise, dim=-1)
        return sample, F.softmax(pred_x_start_logits, dim=-1)
    
    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1,t.cpu()-1)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    # def forward_diffusion(self, x_start, t, mask):
    #     batch_size = x_start.shape[0]
    #     num_classes = 1025
    #     t = t.squeeze()  # Ensure t is of shape (batch_size,)
    #     q_t = self.transition_matrices[t.item()].to("cuda:0")  # Assuming t is the same across the batch

    #     # Convert x_start to one-hot encoding
    #     x_start_one_hot = F.one_hot(x_start, num_classes=num_classes).to(torch.float16)

    #     # Compute x_t probabilities
    #     x_t_probs = torch.matmul(x_start_one_hot, q_t)

    #     # Sample x_t from the categorical distribution defined by x_t_probs
    #     x_t = torch.multinomial(x_t_probs.view(-1, num_classes), num_samples=1).view(x_start.shape)

    #     return x_t*mask

    # def forward_diffusion(self, x_start, t, noise):
    #     """Applies forward diffusion to a sequence of codec indices."""
    #     batch_size = x_start.shape[0]
    #     num_classes = self.num_classes
    #     t = t.squeeze()  # Ensure t is of shape (batch_size,)

    #     # Select the transition matrix for the current timestep
    #     q_t = self.q_onestep_mats[t.item()].to(x_start.device)

    #     # Convert x_start to one-hot encoding
    #     x_start_one_hot = F.one_hot(x_start, num_classes=num_classes).float()

    #     # Compute x_t probabilities
    #     x_t_probs = torch.matmul(x_start_one_hot, q_t)

    #     # Sample x_t from the categorical distribution defined by x_t_probs
    #     x_t = torch.multinomial(x_t_probs.view(-1, num_classes), num_samples=1).view(x_start.shape)

    #     return x_t
    def q_sample(self, x_start, t,mask):
        """Sample from q(x_t | x_start) (i.e. add noise to the data).

        Args:
          x_start: jnp.array: original clean data, in integer form (not onehot).
            shape = (bs, ...).
          t: :jnp.array: timestep of the diffusion process, shape (bs,).
          noise: jnp.ndarray: uniform noise on [0, 1) used to sample noisy data.
            Should be of shape (*x_start.shape, num_pixel_vals).

        Returns:
          sample: jnp.ndarray: same shape as x_start. noisy data.
        """
        noise = torch.rand(size=x_start.shape+(1025,)).to(x_start.device)
        # assert noise.shape == x_start.shape + (self.num_pixel_vals,)
        logits = torch.log(self.q_probs(x_start, t) + self.eps)

        # To avoid numerical issues clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = - torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)*mask
    
    def q_probs(self, x_start, t):
        """Compute probabilities of q(x_t | x_start).

        Args:
          x_start: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
             Should not be of one hot representation, but have integer values
             representing the class values.
          t: jnp.ndarray: jax array of shape (bs,).

        Returns:
          probs: jnp.ndarray: jax array, shape (bs, x_start.shape[1:],
                                                num_pixel_vals).
        """
        return self._at(self.q_mats, t, x_start)
    
    def reverse_diffusion(self, x_t, t, model_fn, noise):
        """Reconstruct the original data from noisy input."""
        # Predict logits from the model for x_{t-1}
        model_logits, pred_x_start_logits = self.p_logits(model_fn=model_fn, x=x_t, t=t)
        
        # Use Gumbel noise to sample from logits
        nonzero_mask = (t != 0).to(x_t.dtype).reshape(x_t.shape[0], *([1] * (len(x_t.shape) - 1)))
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))

        x_t_minus_1 = torch.argmax(model_logits + nonzero_mask * gumbel_noise, dim=-1)

        return x_t_minus_1, F.softmax(pred_x_start_logits, dim=-1)

    def p_logits(self, model_fn, *, x, t):
        """Compute logits of p(x_{t-1} | x_t)."""
        model_output = model_fn(x, t)
        
        if self.model_output == 'logits':
            model_logits = model_output
        else:
            raise NotImplementedError(self.model_output)

        if self.model_prediction == 'x_start':
            pred_x_start_logits = model_logits

            # Adjust logits based on posterior
            t_broadcast = torch.reshape(t, ([model_logits.shape[0]] + [1] * (len(model_logits.shape) - 1)))
            model_logits = torch.where(t_broadcast == 0, pred_x_start_logits, self.q_posterior_logits(pred_x_start_logits, x, t, x_start_logits=True))

        elif self.model_prediction == 'xprev':
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.model_prediction)

        return model_logits, pred_x_start_logits

    # def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
    #     """Compute logits of q(x_{t-1} | x_t, x_start)."""
    #     # if x_start_logits:
    #     #     # assert x_start.shape == x_t.shape + (1025,), (x_start.shape, x_t.shape)
    #     # else:
    #     #     assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

    #     fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
    #     fact2 = self._at(self.q_mats, t, x_start)
    #     return torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)


    
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
        resps_list = torch.stack([resps for resps in resps_list_padded])

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
            if prom_shape < 398:
                # Pad with zeros to make it (200, feature_size), where feature_size is the second dimension
                proms_list_padded.append(F.pad(proms_list[i], (0, 0, 0, 398 - prom_shape)))
            else:
                # If the shape is larger or equal to 200, truncate it to (200, feature_size)
                proms_list_padded.append(proms_list[i][:398, :])

        # Stack the padded tensors back into a tensor
        proms_list = torch.stack(proms_list_padded)
        # Handle conditioning projections for proms_list and text_list
        proms_list = torch.stack([proms[:398, :] for proms in proms_list])




        cond1 = self.proms_emb(proms_list)[0]
        cond1=self.sin_emb.add_pe(cond1)[0]
        cond1 = self.encoder2(cond1).unsqueeze(0)


        cond2 = self.text_emb(text_list)
        cond2 = self.sin_emb.add_pe(cond2)[0]
        cond2 = self.encodertext(cond2).unsqueeze(0)

        
        
        cond = torch.cat([cond1, cond2], dim=1) # Ensure proper concatenation across batch
        # cond = cond.unsqueeze(0)
        # Initialize loss
        self.loss = 0
        print(mask.sum().item())


        
        # Loop over time steps and process each batch sample
        for t in range(1, self.timesteps):
            t_tensor = torch.tensor([t], dtype=torch.long, device="cuda").expand(batch_size)  # Make time tensor batch-compatible
            x_noisy= self.q_sample(resps_list, t_tensor,mask)
            t_emb= self.time_emb(t_tensor)
            x = self.resps_emb(x_noisy)[0]
            x=x*mask.unsqueeze(1)
            # x_noisy = self.sin_emb2.add_pe(x_noisy)[0]

            # x=self.encoder3(x_noisy)
            # x = x_noisy.unsqueeze(1)
            # cond = torch.cat([x,cond,t_emb],dim=0)
            # x_noisy = self.encoder2(x_noisy)
            
            # x=self.token_emb(x)

            # x=self.decout(x,cond)
            # x=x.permute(0,2,1)
            # x=self.conv1d(x)
            # x=x.permute(1,0)
            x=x.unsqueeze(0)
            for block in self.blocks:
                    x = block(x,cond1,cond2,t_emb,mask) 
            
            # x= self.unet(x,t_tensor,encoder_hidden_states=cond,return_dict=True).sample
            # x=x.squeeze(0)
            
            # x= self.final(x)

            # Calculate loss for each batch and accumulate
            
            # x=x.squeeze()
            # x=x.permute(1,0)
            # x=x.squeeze(0)
            x = self.final(x)
            x= x*mask.unsqueeze(1)
            x=x.squeeze()

            # logits = x.view(-1, 1024)
            targets = resps_list.view(-1) * mask
            loss = F.cross_entropy(x, target=targets, reduction='mean')
            # mse_loss = F.mse_loss(x, resps_list.unsqueeze(1)*mask)
            self.loss += loss.mean()
        self.loss=self.loss/mask.sum().item()
        return x

    def generate_audio(self, text_list, proms_list, resps_list=None):
        with torch.no_grad():
            batch_size = len(text_list)  # Assuming all lists have the same length
            resps_list = torch.full(size=[1,350], fill_value=1025 // 2,
                                dtype=torch.int32).to("cuda:0")
            # Pad or truncate `resps_list` to shape [batch_size, 384]
            resps_list_padded = []
            for i in range(batch_size):
                if resps_list[i].shape[0] < 448:
                    resps_list_padded.append(F.pad(resps_list[i], (0, 448 - resps_list[i].shape[0])))
                else:
                    resps_list_padded.append(resps_list[i][:448])
            mask= (resps_list_padded[0]!=0)
            resps_list = torch.stack([resps for resps in resps_list_padded])

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
                if prom_shape < 398:
                    # Pad with zeros to make it (200, feature_size), where feature_size is the second dimension
                    proms_list_padded.append(F.pad(proms_list[i], (0, 0, 0, 398 - prom_shape)))
                else:
                    # If the shape is larger or equal to 200, truncate it to (200, feature_size)
                    proms_list_padded.append(proms_list[i][:398, :])

            # Stack the padded tensors back into a tensor
            proms_list = torch.stack(proms_list_padded)
            cond1 = torch.stack([proms[:398, :] for proms in proms_list])
            cond1 = self.proms_emb(cond1)[0]

            cond2 = self.text_emb(text_list)

            
            cond2 = self.sin_emb.add_pe(cond2)[0]
            cond2 = self.encodertext(cond2).unsqueeze(0)


            cond1=self.sin_emb.add_pe(cond1)[0]
            cond1 = self.encoder2(cond1).unsqueeze(0)

            x_noisy = resps_list
            # Loop over time steps and process each batch sample
            for t in range(self.timesteps-1,0,-1):
                t_tensor = torch.tensor([t], dtype=torch.long, device="cuda").expand(batch_size)  # Make time tensor batch-compatible
                t_emb= self.time_emb(t_tensor)
                x_noise = self.resps_emb(x_noisy)[0]
                t_emb= self.time_emb(t_tensor)
                # x = self.resps_emb(x_noisy)[0]
                # x_noisy = self.encoder2(x_noisy)
                
                # x=self.token_emb(x)

                x_noise=x_noise.unsqueeze(0)
                for block in self.blocks:
                        x_noise = block(x_noise,cond1,cond2,t_emb,mask) 
                x=x_noise
                # x=x.permute(0,2,1)
                # x=self.conv1d(x)
                # x=x.permute(0,1,3,2)
                # x= self.unet(x,t_tensor,encoder_hidden_states=cond,return_dict=True).sample
                # x=x.squeeze(0)
                
                # x= self.final(x)

                # Calculate loss for each batch and accumulate
                x= x[:448,:]*mask.unsqueeze(1)
                # x=x.squeeze()
                # x=x.permute(1,0)
                x_pred= self.final(x)
                # x_pred = torch.argmax(x_pred, dim=2)
                # x_pred=x_pred.squeeze(0)
                x_noisy,_=self.p_sample(x_pred,t_tensor,x_noisy)
            return x_noisy.squeeze()

                
            
     
       
        
