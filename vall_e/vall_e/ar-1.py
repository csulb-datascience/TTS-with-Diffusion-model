# import torch
# from torch import nn, Tensor
# from torch.nn import functional as F
# from torch.nn.utils.rnn import pad_sequence
# from einops import rearrange, repeat
# from tqdm import trange

# class DiffusionModel(nn.Module):
#     def __init__(self, num_qnts, num_steps=1000, d_model=128, nhead=8):
#         super().__init__()
#         self.num_steps = num_steps
#         self.beta = torch.linspace(1e-4, 0.02, num_steps)  # Example noise schedule
#         self.alpha = 1 - self.beta
#         self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
#         self.transformer = UNet(num_qnts)  # Transformer-based UNet

#     def forward(self,x, phoneme_seq, speaker_embedding, max_steps=None):
#         if max_steps is None:
#             max_steps = self.num_steps
        
#         # Ensure inputs are tensors and convert to the same dtype as model parameters
#         phoneme_mask = create_mask(x, 1024).float().to(phoneme_seq[0].device)
#         dtype = next(self.parameters()).dtype
#         phoneme_seq = phoneme_seq[0].to(dtype)
#         speaker_embedding = speaker_embedding[0].to(dtype)
#         x = x[0].to(dtype)
#         phoneme_mask = phoneme_mask.to(dtype)

#         # Initialize the noisy sequence
#         noisy_seq = torch.randn(x.size(0), 1024, dtype=dtype, device=x.device) * phoneme_mask

#         # Reverse process (denoising)
#         for step in reversed(range(max_steps)):
#             noisy_seq = self.denoise_step(noisy_seq, phoneme_seq, speaker_embedding, phoneme_mask, step)
        
#         return noisy_seq 

#     def denoise_step(self, noisy_seq, phoneme_seq, speaker_embedding, phoneme_mask, step):
#         # Denoise the input at each step
#         pred_noise = self.transformer(noisy_seq, phoneme_seq, speaker_embedding, phoneme_mask, step)
#         alpha_step = self.alpha[step]
#         alpha_cumprod_step = self.alpha_cumprod[step]
        
#         noisy_seq = (noisy_seq - pred_noise * (1 - alpha_step).sqrt()) / alpha_step.sqrt()
#         noisy_seq *= phoneme_mask  # Apply mask to ensure padded tokens are not updated
#         return noisy_seq

# # class TransformerUNet(nn.Module):
# #     def __init__(self, num_qnts, d_model, nhead):
# #         super().__init__()
# #         self.encoder = nn.Transformer(d_model=d_model, nhead=nhead)
# #         self.decoder = nn.Transformer(d_model=d_model, nhead=nhead)
# #         self.fc_in = nn.Linear(num_qnts, d_model)
# #         self.fc_out = nn.Linear(d_model, num_qnts)

# #     def forward(self, noisy_seq, phoneme_seq, speaker_embedding, phoneme_mask, step):
# #         x = self.fc_in(noisy_seq)
# #         x = self.encoder(x )#+ phoneme_seq + speaker_embedding)
# #         x = self.decoder(x)
# #         x = self.fc_out(x)
# #         return x * phoneme_mask  # Apply mask to ensure padded tokens are not considered

# class UNet(nn.Module):
#     def __init__(self, num_qnts):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(num_qnts, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv1d(256, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(128, num_qnts, kernel_size=3, padding=1),
#         )

#     def forward(self, x, phoneme_seq, speaker_embedding, step):
#         x = torch.cat([x, phoneme_seq, speaker_embedding], dim=1)
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


# def pad_and_stack(tensor_list, max_length, padding_value=0):
#     padded_list = [F.pad(tensor, (0, max_length - len(tensor)), 'constant', padding_value) for tensor in tensor_list]
#     return torch.stack(padded_list)

# def create_mask(sequence_list, max_length):
#     mask_list = [torch.cat([torch.ones(len(seq)), torch.zeros(max_length - len(seq))]) for seq in sequence_list]
#     return torch.stack(mask_list)

# def example_usage():
#     import soundfile

#     device = "cuda"

#     qnt = torch.load("data/test/p225_002.qnt.pt")[0, 0].to(device)
#     num_qnts = 1024
#     max_length = 1000  # Example fixed maximum length

#     model = DiffusionModel(num_qnts).to(device)

#     text_list = [
#         torch.tensor([1, 2, 3], device=device),
#         torch.tensor([2, 3], device=device),
#     ]

#     phoneme_seq = [
#         torch.tensor([1, 2, 3], device=device),
#         torch.tensor([2, 3], device=device),
#     ]
#     speaker_embedding = [
#         torch.tensor([0.1, 0.2, 0.3], device=device),
#         torch.tensor([0.2, 0.3], device=device),
#     ]

#     # Pad and stack the lists into tensors for batch processing
#     padded_phoneme_seq = pad_and_stack(phoneme_seq, max_length).float()
#     padded_speaker_embedding = pad_and_stack(speaker_embedding, max_length).float()
#     phoneme_mask = create_mask(phoneme_seq, max_length).float()

#     out = model(padded_phoneme_seq, padded_speaker_embedding, phoneme_mask, max_steps=200)

#     print(out)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#     for i in range(100):
#         optimizer.zero_grad()
#         _ = model(padded_phoneme_seq, padded_speaker_embedding, phoneme_mask)

#         losses = model.loss
#         sum(losses.values()).backward()
#         optimizer.step()

#         if i % 20 == 0:
#             print(f"iter={i}, {losses}.")

#     out = model(padded_phoneme_seq, padded_speaker_embedding, phoneme_mask, max_steps=200)

#     print(qnt)
#     print(out)

#     from ..emb.qnt import decode

#     codes = rearrange(out[1], "t -> 1 1 t")
#     wavs, sr = decode(codes)
#     soundfile.write("data/test/test.diffusion.recon.wav", wavs.cpu()[0, 0], sr)

# if __name__ == "__main__":
#     example_usage()




import torch
from torch import nn, Tensor
from einops import rearrange
from tqdm import trange

class DiffusionModel(nn.Module):
    def __init__(self, n_tokens, d_model=512, n_heads=8, n_layers=12, p_dropout=0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.noise_schedule = torch.linspace(0, 1, steps=1000)

        self.text_emb = nn.Embedding(n_tokens, d_model)
        self.proms_emb = nn.Embedding(n_tokens, d_model)
        self.speaker_emb = nn.Linear(d_model, d_model)
        self.sin_emb = SinusoidalEmbedding(d_model)

        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, p_dropout, casual=True)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Linear(d_model, n_tokens + 1)

    def forward_diffusion(self, x, t):
        noise = torch.randn_like(x)
        self.noise_schedule=self.noise_schedule.to(x.device)
        alpha = self.noise_schedule[t]
        return alpha * x + (1 - alpha) * noise, noise

    def reverse_diffusion(self, x, t, condition):
        noise_pred = self.model(x, t, condition)
        alpha = self.noise_schedule[t]
        return (x - (1 - alpha) * noise_pred) / alpha

    def forward(self, text_list, proms_list, resp_list, max_steps=1000):
        device = text_list[0].device
        x = torch.zeros_like(text_list[0]).float().to(device)
        t = torch.randint(0, max_steps, (x.size(0),), device=device)

        for _ in trange(max_steps):
            x, _ = self.forward_diffusion(x, t)
            condition = torch.cat([text_list[0], resp_list[0]], dim=-1)
            x = self.reverse_diffusion(x, t, condition)

        return x

    def _generate(self, text_list, proms_list, speaker_list, max_steps, sampling_temperature):
        device = text_list[0].device
        x = torch.zeros_like(text_list[0]).float().to(device)
        t = torch.randint(0, max_steps, (x.size(0),), device=device)

        for _ in trange(max_steps):
            condition = torch.cat([text_list, proms_list, speaker_list], dim=-1)
            x = self.reverse_diffusion(x, t, condition)

        return x

class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        exponent = torch.arange(self.d_half, dtype=torch.float32)
        exponent = exponent / self.d_half
        omega = torch.exp(-torch.log(torch.tensor(1e4)) * exponent)
        self.register_buffer("omega", omega, persistent=False)

    @property
    def d_half(self):
        assert self.d_model % 2 == 0, "Only support even d_model."
        return self.d_model // 2

    def forward(self, x):
        omega = self.omega

        while omega.dim() <= x.dim():
            omega = omega.unsqueeze(0)  # (... d)

        x = x.unsqueeze(-1)  # (... 1)
        x = omega * x
        x = torch.cat([x.sin(), x.cos()], dim=-1)

        return x

class Block(nn.Module):
    def __init__(self, d_model, n_heads, p_dropout, casual):
        super().__init__()
        dim_head = d_model // n_heads
        self.casual = casual
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, m, l):
        h = x.shape[2] // self.scale
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b t h d', h=h), (q, k, v))

        e = torch.einsum('b i h d, b j h d -> b i j h', q, k) * self.scale

        if self.casual:
            kpm = m.unsqueeze(1) * m.unsqueeze(2)
            kpm = kpm.tril().unsqueeze(-1)
            e = e.masked_fill(kpm == 0, -torch.finfo(e.dtype).max)

        a = e.softmax(dim=2)
        o = torch.einsum('b i j h, b j h d -> b i h d', a, v)
        o = o.flatten(-2)
        o = self.to_out(o)
        o = self.dropout(o)

        return o * m

def example_usage():
    from functools import partial
    from einops import repeat

    device = "cuda"

    qnt = torch.load("data/test/p225_002.qnt.pt")[0, 0].to(device)
    num_qnts = 1024

    model = DiffusionModel(num_qnts).to(device)

    text_list = [
        torch.tensor([1, 2, 3], device=device),
        torch.tensor([2, 3], device=device),
    ]

    x8 = partial(repeat, pattern="t -> t l", l=8)
    proms_list = [
        x8(torch.tensor([1, 2, 3], device=device)),
        x8(torch.tensor([2, 3], device=device)),
    ]

    speaker_list = [
        torch.tensor([0.1, 0.2, 0.3], device=device),
        torch.tensor([0.2, 0.3, 0.4], device=device),
    ]

    out = model(text_list, proms_list, speaker_list, max_steps=200)

    print(out)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(100):
        optimizer.zero_grad()
        _ = model(text_list, proms_list, speaker_list)

        losses = model.loss
        sum(losses.values()).backward()
        optimizer.step()

        if i % 20 == 0:
            print(f"iter={i}, {losses}.")

    out = model(text_list, proms_list, speaker_list, max_steps=200)

    print(qnt)
    print(out)

if __name__ == "__main__":
    example_usage()
