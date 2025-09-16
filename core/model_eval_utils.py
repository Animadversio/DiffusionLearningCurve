"""
Unified model loading function for all models
"""
from os.path import join
import json
import torch
from easydict import EasyDict as edict
from core.DiT_model_lib import DiT
from core.network_edm_lib import SongUNet, create_unet_resnet_model, SongUNetResNet, create_unet_model
from core.diffusion_edm_lib import UNetBlockStyleMLP_backbone
from core.diffusion_edm_lib import EDMPrecondWrapper, EDMCNNPrecondWrapper, EDMDiTPrecondWrapper

def load_pretrained_model(expname, model_type=None, ckpt_name=None, device="cuda"):
    saveroot = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve"
    if model_type is None:
        if "DiT" in expname:
            model_type = "DiT"
        elif "MLP" in expname:
            model_type = "MLP"
        elif "ResNet" in expname:
            model_type = "ResNet"
        elif "UNet_CNN" in expname:
            model_type = "UNet"
        else:
            raise ValueError(f"Model type not found in expname: {expname}")
    if ckpt_name is None:
        ckpt_name = "model_final.pth"
        ckpt_path = join(saveroot, expname, ckpt_name)
    edm_params = {"sigma_data": 0.5, "sigma_min": 0.002, "sigma_max": 80, "rho": 7.0}
    expdir = join(saveroot, expname)
    config = json.load(open(join(expdir, "config.json")))
    if model_type == "MLP":
        from core.diffusion_edm_lib import UNetBlockStyleMLP_backbone, EDMPrecondWrapper
        print("Loading MLP model with config: ", config)
        model = UNetBlockStyleMLP_backbone(**config)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model_precd = EDMPrecondWrapper(model, **edm_params)
    elif model_type == "DiT":
        from core.DiT_model_lib import DiT
        from core.diffusion_edm_lib import UNetBlockStyleMLP_backbone, EDMPrecondWrapper, EDMCNNPrecondWrapper, EDMDiTPrecondWrapper
        print("Loading DiT model with config: ", config)
        model = DiT(**config)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True),)
        model_precd = EDMDiTPrecondWrapper(model, **edm_params)
    elif model_type == "UNet":
        from core.network_edm_lib import SongUNet, DhariwalUNet, create_unet_model
        from core.diffusion_edm_lib import UNetBlockStyleMLP_backbone, EDMPrecondWrapper, EDMCNNPrecondWrapper, EDMDiTPrecondWrapper
        print("Loading CNN UNet model with config: ", config)
        model = create_unet_model(edict(config))
        model.load_state_dict(torch.load(ckpt_path, weights_only=True),)
        model_precd = EDMCNNPrecondWrapper(model, **edm_params)
    elif model_type == "ResNet":
        from core.network_edm_lib import SongUNet, DhariwalUNet, create_unet_resnet_model, SongUNetResNet
        from core.diffusion_edm_lib import UNetBlockStyleMLP_backbone, EDMPrecondWrapper, EDMCNNPrecondWrapper, EDMDiTPrecondWrapper
        print("Loading ResNet model with config: ", config)
        model = create_unet_resnet_model(edict(config))
        model.load_state_dict(torch.load(ckpt_path, weights_only=True),)
        model_precd = EDMCNNPrecondWrapper(model, **edm_params)
    
    # print model parameters
    print(f"Total number of parameters in the model: {sum(p.numel() for p in model.parameters())} | trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model_precd = model_precd.to(device).eval()
    return model_precd, model, config
    

# for expname in ["FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4_fixseed_DSM",
#                 "FFHQ32_UNet_CNN_EDM_4blocks_wide128_attn_pilot_fixednorm_DSM",
#                 "FFHQ32_DiT_P2_384D_6H_6L_EDM_pilot_DSM"]:
#     model_precd, model, config = load_pretrained_model(expname, device="cpu")