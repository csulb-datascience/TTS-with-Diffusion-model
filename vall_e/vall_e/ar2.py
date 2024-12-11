import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .base import MultiEmbedding,Embedding, Base
from torch import Tensor
from torch.distributions import Categorical
from tqdm import trange
import numpy as np
from diffusers import UNet3DConditionModel, UNet2DConditionModel
# from transformers import Transformer

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
        # Define the embeddings
        self.text_emb = Embedding(n_tokens, d_model)
        self.proms_emb = MultiEmbedding(max_n_levels, n_tokens, d_model)
        self.resps_emb = MultiEmbedding(1, n_tokens, d_model)
        self.time_emb = Embedding(101, d_model)

        self.positional_encoding = PositionalEncoding(d_model)


        # # Define the attention layer
        # self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        # self.resp_encoder = nn.TransformerEncoder(d_model=d_model, nhead=n_heads,num_layers=2) 
        # self.prom_encoder = nn.TransformerEncoder(d_model=d_model, nhead=n_heads,num_layers=2) 
        # self.encoder = nn.TransformerEncoder(d_model=d_model, nhead=n_heads,num_layers=2) 
        # self.xencoder = nn.TransformerEncoder(d_model=d_model, nhead=n_heads,num_layers=2) 

        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads),
            num_layers=1
        )
        self.proms_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads),
            num_layers=1
        )
        self.noise_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads),
            num_layers=1
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads),
            num_layers=1
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads),
            num_layers=6
        )
        self.output_layer = nn.Linear(d_model, d_model)




        # # FiLM layers for conditioning
        # self.film1 = FiLM(d_model)
        # self.film2 = FiLM(d_model)
        # self.film3 = FiLM(d_model)
        # in_channels=1
        # num_filters=64
        # out_channels=1
        # # Downsample (Encoder)
        # self.down1 = nn.Sequential(
        #     nn.Conv1d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     # nn.BatchNorm1d(num_filters)
        # )
        # self.down2 = nn.Sequential(
        #     nn.Conv1d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(num_filters * 2)
        # )
        # self.down3 = nn.Sequential(
        #     nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(num_filters * 4)
        # )

        # # Bottleneck
        # self.bottleneck = nn.Sequential(
        #     nn.Conv1d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(num_filters * 8)
        # )

        # # Upsample (Decoder)
        # self.up1 = nn.Sequential(
        #     nn.ConvTranspose1d(num_filters * 8, num_filters * 4, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(num_filters * 4)
        # )
        # self.up2 = nn.Sequential(
        #     nn.ConvTranspose1d(num_filters * 4, num_filters * 2, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(num_filters * 2)
        # )
        # self.up3 = nn.Sequential(
        #     nn.ConvTranspose1d(num_filters * 2, num_filters, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(num_filters)
        # )

        # # Final output layer
        # self.final = nn.ConvTranspose1d(num_filters, out_channels, kernel_size=4, stride=2, padding=1)
        self.num_steps=self.timesteps = 100
        self.beta_start = 0.002
        # self.beta_end = 0.02
        
        # Define beta schedule
        # Define beta schedule
        self.betas = self.cosine_beta_schedule(self.timesteps, self.beta_start).to(torch.float16)

        # Pre-calculate different terms for closed form
        self.alphas = (1.0 - self.betas).to(torch.float16)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(torch.float16)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(torch.float16)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(torch.float16)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(torch.float16)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(torch.float16)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).to(torch.float16)

    
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
        x_0=x_0[0]
        noise = torch.randn_like(x_0).to(device="cuda")
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        x_noisy = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
                  + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        return x_noisy, noise
    
    def reverse_diffusion(self, x_noisy, t, noise, device="cuda"):
        """ 
        Reconstruct the original image from the noisy input.
        """
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x_noisy.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
        
        # Mean reconstruction
        x_pred = sqrt_recip_alphas_t.to(device) * (x_noisy.to(device) - sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device))
        
        return x_pred

        

        

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
        text_emb = self.text_emb(text_list)
        proms_emb = self.proms_emb(proms_list)
        resps_emb = self.resps_emb(_unsqueeze_list(resps_list))

        # text_emb = self.positional_encoding(text_emb[0])
        # proms_emb = self.positional_encoding(proms_emb[0])
        # resps_emb = self.positional_encoding(resps_emb[0])
        # embeddings = [torch.cat([t, p], dim=0) for t, p in zip(text_emb, proms_emb)]
        # x, m = list_to_tensor(embeddings)  # (batch_size, sequence_length, d_model)

        self.loss = 0
        if resps_list is not None:
            for t in range(1, self.num_steps+1 ):
                t_tensor = torch.tensor([t], dtype=torch.long, device="cuda")
                x_noisy, noise = self.forward_diffusion(resps_emb, t_tensor)
                t_emb = self.time_emb([t_tensor])
                # embeddings = [torch.cat([t, p], dim=0) for t, p in zip(text_emb, proms_emb)]
                # x, m = list_to_tensor(embeddings)  # (batch_size, sequence_length, d_model)
                # x_pred = self.encoder(embeddings[0],x_noisy[0])
                text_context = self.text_encoder(text_emb[0])  # Shape [a, 1024]
                proms_context = self.proms_encoder(proms_emb[0])  # Shape [b, 1024]
                noise_context = self.noise_encoder(x_noisy)  # Shape [c, 1024]
                  # Shape [d, 1024]
                # combined_features = self.attention(text_context,proms_context,x_noisy[0])
                # # Pooling to get fixed-size representations
                # text_pooled = torch.mean(text_context, dim=0)  # Shape [1024]
                # proms_pooled = torch.mean(proms_context, dim=0)  # Shape [1024]
                # noise_pooled = torch.mean(noise_context, dim=0)  # Shape [1024]
                # time_pooled = torch.mean(time_context, dim=0)  # Shape [1024]

                # Combine the pooled features
                combined_features = torch.cat([text_context,proms_context,t_emb[0]], dim=0) 



                x_pred=self.decoder(noise_context,combined_features)
                x_pred = self.encoder(x_pred)
                # Final output layer to predict the noise
                x_pred = self.output_layer(x_pred)
                # input=x_noisy[0].unsqueeze(0).unsqueeze(0)
                # encoded_hidden_state=embeddings[0].unsqueeze(0)
                # x_pred = self.Unet(input,t,encoded_hidden_state)
                # x_pred = self.output_layer(x_pred)
                # cosine_loss = 1 - F.cosine_similarity(x_pred, noise).mean()
                mse_loss=F.mse_loss(x_pred, noise)
                # l1_loss=F.l1_loss(x_pred,noise)
                self.loss += mse_loss#1*mse_loss#+0*cosine_loss
            return noise
    # def noise_schedule(self, t):
    #     return torch.tensor(self.beta_start + t * (self.beta_end - self.beta_start)/(self.num_steps-1))

    # def forward_diffusion(self, x, t):
    #     noise = torch.randn_like(x[0])
    #     beta = self.noise_schedule(t)
    #     alpha_t = 1 - beta
    #     alpha_bar_t = torch.prod(alpha_t)  # Alternatively, use cumulative product over timesteps if needed
    #     x_noisy = x[0] * torch.sqrt(alpha_bar_t) + noise * torch.sqrt(1 - alpha_bar_t)
    #     return [x_noisy], noise


    # def reverse_diffusion(self, x_noisy, noise, beta, t):
    #     """
    #     Args:
    #     - x_noisy: The noisy input at timestep t.
    #     - noise: The predicted noise at timestep t.
    #     - beta: The beta value at timestep t.
    #     - t: The current timestep.

    #     Returns:
    #     - x: The denoised output at timestep t-1.
    #     """
    #     alpha_t = 1 - beta
    #     alpha_bar_t = torch.prod(alpha_t)  # Use cumulative product up to t
    #     alpha_bar_t_prev = torch.prod(alpha_t[:t]) if t > 0 else torch.tensor(1.0).to(x_noisy.device)
        
    #     x_pred = (x_noisy - noise * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t)
        
    #     # Add noise at each step except the last one
    #     if t > 0:
    #         noise_t = torch.randn_like(x_noisy)
    #         x_pred = x_pred + torch.sqrt(beta) * noise_t
        
    #     return x_pred

    def generate_audio(self, text_list, proms_list, resps_list=None):
        with torch.no_grad():
            resps_list = [torch.randint(low=0, high=1024, size=(200,), dtype=torch.long,device="cuda")]
            text_emb = self.text_emb(text_list)
            proms_emb = self.proms_emb(proms_list)
            resps_emb = self.resps_emb(_unsqueeze_list(resps_list))
            # resp = proms_list[0][:, [0]].squeeze(1)
            # resp_emp = self.resps_emb(_unsqueeze_list([resp]))
            # resp_emp1 =self.resps_emb(_unsqueeze_list([resp]))
            # ox=resp_emp[0]-resp_emp1[0]
            
            # x, m = list_to_tensor(embeddings)  # (batch_size, sequence_length, d_model)
            x_noisy = resps_emb[0]
            for t in range(self.timesteps, 0, -1):
                t_tensor = t_tensor = torch.tensor([t], dtype=torch.long, device="cuda")
                t_emb = self.time_emb([t_tensor])
                text_context = self.text_encoder(text_emb[0])  # Shape [a, 1024]
                proms_context = self.proms_encoder(proms_emb[0])  # Shape [b, 1024]
                noise_context = self.noise_encoder(x_noisy)  # Shape [c, 1024]
                # time_context = self.noise_encoder(t_emb[0]) 
                combined_features = torch.cat([text_context,proms_context,t_emb[0]], dim=0) 



                x_pred=self.decoder(noise_context,combined_features)
                x_pred = self.encoder(x_pred)
                # Final output layer to predict the noise
                x_pred = self.output_layer(x_pred)
                
                x_noisy = self.reverse_diffusion(x_noisy,t_tensor, x_pred)       

            # for t in range(100, 0, -1):
            #     x_pred = self.encoder(embeddings[0],x_noisy)
            #     x_pred = self.output_layer(x_pred)
            #     beta = self.noise_schedule(t)
            #     x_noisy = self.reverse_diffusion(x_noisy, x_pred, beta)

            # for t in range(20, 0, -1):
            #     x_pred = self.encoder(embeddings[0],x_noisy)
            #     x_pred = self.output_layer(x_pred)
            #     beta = self.noise_schedule(t)
            #     x_noisy = self.reverse_diffusion(x_noisy, x_pred, beta)

            ret = self.find_closest_embedding(x_noisy,self.resps_emb)
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
