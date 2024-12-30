import torch
import os
import logging
import folder_paths
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
import numpy as np
from .attention_processor_advanced import IPAFluxAttnProcessor2_0Advanced
from .utils import is_model_patched, FluxUpdateModules

MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter-flux")
if "ipadapter-flux" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter-flux"]
folder_paths.folder_names_and_paths["ipadapter-flux"] = (current_paths, folder_paths.supported_pt_extensions)

class MLPProjModelAdvanced(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class InstantXFluxIPAdapterModelAdvanced:
    def __init__(self, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        # load image encoder
        self.image_encoder = SiglipVisionModel.from_pretrained(self.image_encoder_path).to(self.device, dtype=torch.float16)
        self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
        # state_dict
        self.state_dict = torch.load(os.path.join(MODELS_DIR,self.ip_ckpt), map_location="cpu")
        self.joint_attention_dim = 4096
        self.hidden_size = 3072

    def init_proj(self):
        self.image_proj_model = MLPProjModelAdvanced(
            cross_attention_dim=self.joint_attention_dim, # 4096
            id_embeddings_dim=1152, 
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)

    def set_ip_adapter_advanced(self, flux_model, weight_params, timestep_percent_range=(0.0, 1.0)):
        weight_start, weight_end, steps = weight_params
        s = flux_model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))
        ip_attn_procs = {}
        dsb_count = len(flux_model.diffusion_model.double_blocks)
        for i in range(dsb_count):
            name = f"double_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0Advanced(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale_start=weight_start,
                    scale_end=weight_end,
                    total_steps=steps,
                    timestep_range=timestep_range
                ).to(self.device, dtype=torch.float16)
        ssb_count = len(flux_model.diffusion_model.single_blocks)
        for i in range(ssb_count):
            name = f"single_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0Advanced(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale_start=weight_start,
                    scale_end=weight_end,
                    total_steps=steps,
                    timestep_range=timestep_range
                ).to(self.device, dtype=torch.float16)
        return ip_attn_procs
    
    def load_ip_adapter_advanced(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        self.image_proj_model.load_state_dict(self.state_dict["image_proj"], strict=True)
        ip_attn_procs = self.set_ip_adapter_advanced(flux_model, weight, timestep_percent_range)
        ip_layers = torch.nn.ModuleList(ip_attn_procs.values())
        ip_layers.load_state_dict(self.state_dict["ip_adapter"], strict=True)
        return ip_attn_procs

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
            clip_image_embeds = clip_image_embeds.to(dtype=torch.float16)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds

class IPAdapterFluxLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter-flux"),),
                "clip_vision": (["google/siglip-so400m-patch14-384"],),
                "provider": (["cuda", "cpu", "mps"],),
            }
        }
    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    RETURN_NAMES = ("ipadapterFlux",)
    FUNCTION = "load_model_advanced"
    CATEGORY = "InstantXNodes"

    def load_model_advanced(self, ipadapter, clip_vision, provider):
        logging.info("Loading InstantX IPAdapter Flux model.")
        model = InstantXFluxIPAdapterModelAdvanced(image_encoder_path=clip_vision, ip_ckpt=ipadapter, device=provider, num_tokens=128)
        return (model,)

class ApplyIPAdapterFluxAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter_flux": ("IP_ADAPTER_FLUX_INSTANTX", ),
                "image": ("IMAGE", ),
                "weight_start": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "weight_end": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter_flux_advanced"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter_flux_advanced(self, model, ipadapter_flux, image, weight_start, weight_end, steps, start_percent, end_percent):
        # Clean up old processors if they exist
        if hasattr(model.model, '_ip_attn_procs'):
            for proc in model.model._ip_attn_procs.values():
                proc.clear_memory()  # Add a new method for cleanup
            del model.model._ip_attn_procs

        pil_image = image.numpy()[0] * 255.0
        pil_image = Image.fromarray(pil_image.astype(np.uint8))
        ipadapter_flux.init_proj()
        
        IPAFluxAttnProcessor2_0Advanced.reset_all_instances()
        
        ip_attn_procs = ipadapter_flux.load_ip_adapter_advanced(model.model, (weight_start, weight_end, steps), (start_percent, end_percent))
        
        image_prompt_embeds = ipadapter_flux.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=None
        )
        is_patched = is_model_patched(model.model)
        bi = model.clone()
        FluxUpdateModules(bi, ip_attn_procs, image_prompt_embeds, is_patched)
        
        # Store reference to processors for cleanup
        bi.model._ip_attn_procs = ip_attn_procs
        
        return (bi,)

NODE_CLASS_MAPPINGS = {
    "IPAdapterFluxLoaderAdvanced": IPAdapterFluxLoaderAdvanced,
    "ApplyIPAdapterFluxAdvanced": ApplyIPAdapterFluxAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterFluxLoaderAdvanced": "Load IPAdapter Flux Model (Advanced)",
    "ApplyIPAdapterFluxAdvanced": "Apply IPAdapter Flux Model (Advanced)",
}

