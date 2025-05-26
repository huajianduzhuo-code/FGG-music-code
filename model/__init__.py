from .latent_diffusion import LatentDiffusion
from .model_sdf import Diffpro_SDF
from .architecture.unet import UNetModel
import os


def init_ldm_model(params, debug_mode=False):
    unet_model = UNetModel(
       in_channels=params.in_channels,
       out_channels=params.out_channels,
       channels=params.channels,
       attention_levels=params.attention_levels,
       n_res_blocks=params.n_res_blocks,
       channel_multipliers=params.channel_multipliers,
       n_heads=params.n_heads,
       tf_layers=params.tf_layers,
    )


    ldm_model = LatentDiffusion(
        unet_model=unet_model,
        latent_scaling_factor=params.latent_scaling_factor,
        n_steps=params.n_steps,
        linear_start=params.linear_start,
        linear_end=params.linear_end,
        debug_mode=debug_mode
    )

    return ldm_model



def init_diff_pro_sdf(ldm_model, params, device):
    return Diffpro_SDF(ldm_model).to(device)


def get_model_path(model_dir, model_id=None):
    model_desc = os.path.basename(model_dir)
    if model_id is None:
        model_path = os.path.join(model_dir, 'chkpts', 'weights.pt')

        # retrieve real model_id from the actual file weights.pt is pointing to
        model_id = os.path.basename(os.path.realpath(model_path)).split('-')[1].split('.')[0]

    elif model_id == 'best':
        model_path = os.path.join(model_dir, 'chkpts', 'weights_best.pt')
        # retrieve real model_id from the actual file weights.pt is pointing to
        model_id = os.path.basename(os.path.realpath(model_path)).split('-')[1].split('.')[0]
    elif model_id == 'default':
        model_path = os.path.join(model_dir, 'chkpts', 'weights_default.pt')
        if not os.path.exists(model_path):
            return get_model_path(model_dir, 'best')
    else:
        model_path = f"{model_dir}/chkpts/weights-{model_id}.pt"
    return model_path, model_id, model_desc
