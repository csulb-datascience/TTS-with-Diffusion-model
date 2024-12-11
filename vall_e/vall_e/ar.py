import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from tqdm import trange
import numpy as np
from .base import Base
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
    

class AR(Base):
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

    @staticmethod
    def _unsqueeze_list(x_list, axis=-1):
        return [x.unsqueeze(dim=axis) for x in x_list]

    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resp_list: list[Tensor] | None = None,
        max_steps: int = 1000,
        sampling_temperature: float = 1.0,
    ):
        if resp_list is not None:
            return super().forward(
                text_list,
                proms_list,
                self._unsqueeze_list(resp_list),
                resp_list,
                quant_levels=None,
                shift_targ_list=True,
                return_all_resp=False,
            )
        else:
            return self._generate(
                text_list,
                proms_list,
                max_steps,
                sampling_temperature,
            )

    def _generate(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        max_steps: int,
        sampling_temperature: float,
    ):
        device = text_list[0].device
        resp_list: list[Tensor] = [
            torch.zeros(0, device=device).long() for _ in text_list
        ]
        stopped = torch.zeros(len(text_list), device=device).bool()
        for _ in range(max_steps):
            r = super().forward(
                text_list,
                proms_list,
                self._unsqueeze_list(resp_list),
                sampling_temperature=sampling_temperature,
            )
            stopped |= r == self.stop_token
            for i, ri in enumerate(r):
                resp_list[i] = torch.cat([resp_list[i], ri[None]])
            if stopped.all().item():
                break
        pruned = [self._prune(r) for r in resp_list]
        # pruned=self.pad_or_slice_tensor(pruned)
        return pruned
    def vpsde_beta_t(self,t, T, min_beta, max_beta):
        t_coef = (2 * t - 1) / (T ** 2)
        return 1. - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)
    def vspe_beta_schedule(self, timesteps,min_beta=0.1,max_beta=40):
        schedule_list = np.array([
            self.vpsde_beta_t(t, timesteps, min_beta, max_beta) for t in range(1, timesteps + 1)])
        return torch.from_numpy(schedule_list)
    def pad_or_slice_tensor(self,resps_list, max_size=448):
        batch_size = len(resps_list)
        resps_list_padded = []
        for i in range(batch_size):
            if resps_list[i].shape[0] < 448:
                resps_list_padded.append(F.pad(resps_list[i], (0, 448 - resps_list[i].shape[0])))
            else:
                resps_list_padded.append(resps_list[i][:448])
        # mask= (resps_list_padded[0]!=0)
        resps_list = [resps for resps in resps_list_padded]
        return resps_list
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


# def example_usage():
#     from functools import partial

#     import soundfile
#     from einops import repeat

#     device = "cuda"

#     qnt = torch.load("data/test/.qnt.pt")[0, 0].to(device)
#     num_qnts = 1024

#     model = AR(num_qnts).to(device)

#     text_list = [
#         torch.tensor([1, 2, 3], device=device),
#         torch.tensor([2, 3], device=device),
#     ]

#     x8 = partial(repeat, pattern="t -> t l", l=8)
#     proms_list = [
#         x8(torch.tensor([1, 2, 3], device=device)),
#         x8(torch.tensor([2, 3], device=device)),
#     ]

#     resp_list = [
#         torch.tensor([1, 2, 3], device=device),
#         qnt.to(device),
#     ]

#     out = model(text_list, proms_list, max_steps=200)

#     print(out)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#     for i in range(100):
#         optimizer.zero_grad()
#         _ = model(text_list, proms_list, resp_list)

#         losses = model.loss
#         sum(losses.values()).backward()
#         optimizer.step()

#         if i % 20 == 0:
#             print(f"iter={i}, {losses}.")

#     out = model(text_list, proms_list, max_steps=200)

#     print(qnt)
#     print(out)

#     from ..emb.qnt import decode

#     codes = rearrange(out[1], "t -> 1 1 t")
#     wavs, sr = decode(codes)
#     soundfile.write("data/test/test.ar.recon.wav", wavs.cpu()[0, 0], sr)


# if __name__ == "__main__":
#     example_usage()
