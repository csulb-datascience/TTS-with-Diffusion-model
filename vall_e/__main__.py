import argparse
from pathlib import Path

import torch
from einops import rearrange

from .emb import g2p, qnt
from .utils import to_device


def main():
    # parser = argparse.ArgumentParser("VALL-E TTS")
    # parser.add_argument("text")
    # parser.add_argument("reference", type=Path)
    # parser.add_argument("out_path", type=Path)
    # parser.add_argument("--ar-ckpt", type=Path, default="zoo/ar.pt")
    # parser.add_argument("--nar-ckpt", type=Path, default="zoo/nar.pt")
    # parser.add_argument("--device", default="cuda")
    # args = parser.parse_args()

    # ar = torch.load(args.ar_ckpt).to(args.device)
    # nar = torch.load(args.nar_ckpt).to(args.device)

    # symmap = ar.phone_symmap

    # proms = qnt.encode_from_file(args.reference)
    # proms = rearrange(proms, "1 l t -> t l")

    # phns = torch.tensor([symmap[p] for p in g2p.encode(args.text)])

    # proms = to_device(proms, args.device)
    # phns = to_device(phns, args.device)
    # # noise_level = torch.randint(0, 100,(1,), device="cuda")
    # # resps_list = proms
    # resp_list = ar.generate_audio(text_list=[phns], proms_list=[proms])
    # resps_list = [r.unsqueeze(-1) for r in [resp_list]]
    # # resp = resps_list[0][..., 0]
    # resps_list = nar(text_list=[phns], proms_list=[proms], resps_list=resps_list)
    # # resps_list=resps_list
    # # p=proms.clone()
    # # p[:,0]=resp_list
    # qnt.decode_to_file(resps=resps_list[0], path=args.out_path)
    # print(args.out_path, "saved.")
    parser = argparse.ArgumentParser("VALL-E TTS")
    parser.add_argument("text")
    parser.add_argument("reference", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--ar-ckpt", type=Path, default="zoo/ar.pt")
    parser.add_argument("--nar-ckpt", type=Path, default="zoo/nar.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ar = torch.load(args.ar_ckpt).to(args.device)
    nar = torch.load(args.nar_ckpt).to(args.device)

    symmap = ar.phone_symmap

    proms = qnt.encode_from_file(args.reference)
    proms = rearrange(proms, "1 l t -> t l")

    phns = torch.tensor([symmap[p] for p in g2p.encode(args.text)])

    proms = to_device(proms, args.device)
    phns = to_device(phns, args.device)

    resps_list = proms
    resp_list = ar(text_list=[phns], proms_list=[proms])
    resps_list = [r.unsqueeze(-1) for r in resp_list]


    resps_list = nar(text_list=[phns], proms_list=[proms], resps_list=resps_list)
    qnt.decode_to_file(resps=resps_list[0], path=args.out_path)
    print(args.out_path, "saved.")


if __name__ == "__main__":
    main()
