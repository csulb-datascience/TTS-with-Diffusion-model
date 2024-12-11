from ..config import cfg
from .ar import AR
from .nar import NAR
# from .diffused_ar import DifusionTTS


def get_model(name):
    name = name.lower()

    if name.startswith("ar"):
        Model = AR
        # max_n_levels = 8
        # n_tokens = 1024
        # d_model = 1024
        # n_steps = 50
        # n_heads = 8
        # num_layers = 8

        # return Model(d_model=d_model,n_tokens=n_tokens,n_steps=n_steps, max_n_levels=max_n_levels, n_heads=n_heads, num_layers=num_layers).to("cuda")
    elif name.startswith("nar"):
        Model = NAR
    elif name.startswith("diffusion"):
        Model = AR
        max_n_levels = 8
        n_tokens = 1024
        d_model = 512
        n_steps = 100
        n_heads = 8
        num_layers = 6

        return Model(d_model, n_steps, n_tokens, max_n_levels, n_heads, num_layers).to("cuda")
    else:
        raise ValueError("Model name should start with AR or NAR.")

    if "-quarter" in name:
        model = Model(
            cfg.num_tokens,
            d_model=256,
            n_heads=4,
            n_layers=12,
        )
    elif "-half" in name:
        model = Model(
            cfg.num_tokens,
            d_model=512,
            n_heads=8,
            n_layers=12,
        )
    else:
        if name not in ["ar", "nar", "diffusion"]:
            raise NotImplementedError(name)
        model = Model(
            cfg.num_tokens,
            d_model=1024,
            n_heads=16,
            n_layers=12,
        )

    return model
