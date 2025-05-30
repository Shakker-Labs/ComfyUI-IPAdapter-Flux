import torch
import os
import logging
import folder_paths
import json
from transformers import AutoProcessor, CLIPVisionModel, SiglipVisionModel
from PIL import Image
import numpy as np
from .attention_processor import IPAFluxAttnProcessor2_0
from .utils import is_model_patched, FluxUpdateModules, _compute_sha256
from safetensors.torch import load_file
from .hashes import KNOWN_XLAB_HASHES

DEBUG_ENABLED = False

# Define model directories
MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter-flux")
CLIP_VISION_DIR = os.path.join(folder_paths.models_dir, "clip_vision")
if "ipadapter-flux" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter-flux"]
    MODELS_DIR = current_paths[0]

folder_paths.folder_names_and_paths["ipadapter-flux"] = (current_paths, folder_paths.supported_pt_extensions)

# Mapping of file names to HF models
CLIP_MODEL_MAPPING = {
    "clip-vit-large-patch14.safetensors": "openai/clip-vit-large-patch14",
    "siglip-so400m-patch14-384.safetensors": "google/siglip-so400m-patch14-384",
    "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
}

# Scan clip_vision directory for models
def get_clip_vision_models():
    clip_vision_models = []
    local_model_names = set()

    for root, dirs, files in os.walk(CLIP_VISION_DIR):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            has_config = os.path.exists(os.path.join(dir_path, "config.json"))
            has_model = os.path.exists(os.path.join(dir_path, "pytorch_model.bin")) or os.path.exists(os.path.join(dir_path, "model.safetensors"))
            if has_config and has_model:
                relative_path = os.path.relpath(dir_path, CLIP_VISION_DIR).replace(os.sep, "/")
                clip_vision_models.append(relative_path)
                model_name = relative_path.split("/")[-1]
                local_model_names.add(model_name)
                logging.info(f"Found valid CLIP Vision directory: {relative_path}")
        for file_name in files:
            if file_name.endswith(".safetensors"):
                relative_path = os.path.relpath(os.path.join(root, file_name), CLIP_VISION_DIR).replace(os.sep, "/")
                clip_vision_models.append(relative_path)
                model_name = file_name.replace(".safetensors", "")
                local_model_names.add(model_name)
                logging.info(f"Found CLIP Vision safetensors file: {relative_path}")

    hf_models = [
        "google/siglip-so400m-patch14-384",
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-base-patch32"
    ]
    for hf_model in hf_models:
        model_name = hf_model.split("/")[-1]
        if model_name not in local_model_names:
            clip_vision_models.append(hf_model)
            logging.info(f"Added HF model to list: {hf_model}")
        else:
            logging.info(f"Skipped HF model {hf_model} due to local version: {model_name}")

    unique_models = sorted(list(set(clip_vision_models)))
    logging.info(f"Available CLIP Vision models: {unique_models}")
    return unique_models

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4, single_layer=False):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.id_embeddings_dim = id_embeddings_dim
        self.num_tokens = num_tokens
        logging.debug(f"MLPProjModel init: id_embeddings_dim={id_embeddings_dim}, num_tokens={num_tokens}, single_layer={single_layer}")
        if single_layer:
            # One layer projection for XLabs
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(id_embeddings_dim, cross_attention_dim * num_tokens),
            )
        else:
            # Two-layer projection for other cases (InstantX)
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
                torch.nn.GELU(),
                torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
            )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class InstantXFluxIPAdapterModel:
    def __init__(self, image_encoder_path, ip_ckpt, device, dtype=torch.bfloat16, shuffle_weights=False):
        self.device = device
        self.dtype = dtype
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.fallback_proj = None

        # Load Vision model
        try:
            clip_path = os.path.join(CLIP_VISION_DIR, image_encoder_path.replace("/", os.sep))
            if os.path.exists(clip_path):
                logging.info(f"Loading CLIP Vision from local path: {clip_path}")
                if clip_path.endswith(".safetensors"):
                    state_dict = load_file(clip_path, device="cpu")
                    # Determine hf_model dynamically
                    model_name = os.path.basename(clip_path)
                    hf_model = CLIP_MODEL_MAPPING.get(model_name)
                    if not hf_model:
                        # Check config.json in the same directory
                        config_path = os.path.join(os.path.dirname(clip_path), "config.json")
                        if os.path.exists(config_path):
                            with open(config_path, "r") as f:
                                config = json.load(f)
                                # Try multiple keys for model identification
                                hf_model = config.get("_name_or_path") or config.get("model_type") or config.get("architecture")
                                if hf_model:
                                    logging.info(f"Extracted hf_model from config.json: {hf_model}")
                                else:
                                    logging.warning(f"No valid model identifier in {config_path}. Checking local fallback model.")
                                    # Check for local clip-vit-large-patch14.safetensors
                                    fallback_model = "clip-vit-large-patch14.safetensors"
                                    fallback_path = os.path.join(CLIP_VISION_DIR, fallback_model)
                                    if os.path.exists(fallback_path) and fallback_model in CLIP_MODEL_MAPPING:
                                        hf_model = CLIP_MODEL_MAPPING[fallback_model]
                                        clip_path = fallback_path
                                        logging.info(f"Using local model: {clip_path} with hf_model: {hf_model}")
                                    else:
                                        hf_model = "openai/clip-vit-large-patch14"
                                        logging.info(f"Falling back to Hugging Face model: {hf_model}")
                        else:
                            # No config.json, check local fallback model
                            logging.warning(f"No config.json for {model_name}. Checking local fallback model.")
                            fallback_model = "clip-vit-large-patch14.safetensors"
                            fallback_path = os.path.join(CLIP_VISION_DIR, fallback_model)
                            if os.path.exists(fallback_path) and fallback_model in CLIP_MODEL_MAPPING:
                                hf_model = CLIP_MODEL_MAPPING[fallback_model]
                                clip_path = fallback_path
                                logging.info(f"Using local model: {clip_path} with hf_model: {hf_model}")
                            else:
                                if "/" in image_encoder_path and not os.path.exists(image_encoder_path):
                                    hf_model = image_encoder_path
                                    logging.info(f"Using image_encoder_path as hf_model: {hf_model}")
                                else:
                                    hf_model = "openai/clip-vit-large-patch14"
                                    logging.warning(f"No local fallback model found. Falling back to Hugging Face model: {hf_model}")

                    # Selecting model class
                    if "siglip" in model_name.lower() or "siglip" in hf_model.lower():
                        logging.info(f"Loading SiglipVisionModel for: {hf_model}")
                        self.image_encoder = SiglipVisionModel.from_pretrained(
                            hf_model, state_dict=None, ignore_mismatched_sizes=True
                        ).to(self.device, dtype=self.dtype)
                    else:
                        logging.info(f"Loading CLIPVisionModel for: {hf_model}")
                        self.image_encoder = CLIPVisionModel.from_pretrained(
                            hf_model, state_dict=None, ignore_mismatched_sizes=True
                        ).to(self.device, dtype=self.dtype)

                    self.image_encoder.load_state_dict(state_dict, strict=False)
                    self.clip_image_processor = AutoProcessor.from_pretrained(hf_model)
                else:
                    self.image_encoder = CLIPVisionModel.from_pretrained(clip_path).to(self.device, dtype=self.dtype)
                    self.clip_image_processor = AutoProcessor.from_pretrained(clip_path)
            else:
                logging.info(f"Loading CLIP Vision from Hugging Face: {image_encoder_path}")
                if "siglip" in image_encoder_path.lower():
                    self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=self.dtype)
                else:
                    self.image_encoder = CLIPVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=self.dtype)
                self.clip_image_processor = AutoProcessor.from_pretrained(image_encoder_path)
        except Exception as e:
            logging.error(f"Failed to load CLIP Vision: {e}")
            raise

        # Loading IP-Adapter
        path = os.path.join(MODELS_DIR, self.ip_ckpt)
        logging.info(f"Loading IP-Adapter: {path}")
        if path.endswith('.safetensors'):
            state_dict = load_file(path, device="cpu")
            logging.info("Loaded using safetensors")
        else:
            state_dict = torch.load(path, map_location="cpu")
            logging.info("Loaded using torch.load")

        sha256_hash = _compute_sha256(path)
        is_xlab = sha256_hash in KNOWN_XLAB_HASHES
        self.is_xlab = is_xlab  # Store is_xlab for init_ip_adapter
        logging.info(f"SHA256: {sha256_hash}, is_xlab: {is_xlab}")

        self.joint_attention_dim = 4096
        self.hidden_size = 3072

        # Log state_dict keys
        all_keys = list(state_dict.keys())
        logging.debug(f"State dict keys (first 10): {all_keys[:10]}")

        # Determine num_tokens
        self.num_tokens = 4
        logging.info(f"Set default num_tokens={self.num_tokens}")

        # Find image_proj keys
        image_proj_dict = state_dict.get("image_proj", {})
        if not image_proj_dict:
            # Try ip_adapter for proj keys (InstantX)
            image_proj_dict = {k.replace("ip_adapter.", ""): v for k, v in state_dict.items() if k.startswith("ip_adapter.") and "proj" in k.lower()}
            if image_proj_dict:
                logging.debug(f"Found ip_adapter proj keys: {list(image_proj_dict.keys())[:5]}")
            else:
                logging.warning("No image_proj, InstantX, or proj keys found")

        # Detection based on state_dict or file name
        if not is_xlab and (any(k.startswith("double_blocks") for k in state_dict) or "flux" in self.ip_ckpt.lower()):
            self.num_tokens = 128
            logging.info(f"Fallback to Flux.1: num_tokens={self.num_tokens}")
        elif "sdxl" in self.ip_ckpt.lower():
            raise ValueError("SDXL is not supported")
        elif "sd15" in self.ip_ckpt.lower():
            raise ValueError("SD 1.5 is not supported")

        # Auto-detect num_tokens as fail-safe
        if "norm.weight" in image_proj_dict and ("proj.weight" in image_proj_dict or "proj.2.weight" in image_proj_dict):
            norm_size = image_proj_dict["norm.weight"].shape[0]
            if "proj.weight" in image_proj_dict:  # Single-layer projector
                weight_shape = image_proj_dict["proj.weight"].shape
                if weight_shape[0] % norm_size == 0:
                    self.num_tokens = weight_shape[0] // norm_size
                    logging.info(f"Detected num_tokens={self.num_tokens} from proj.weight shape={weight_shape}, norm_size={norm_size}")
            elif "proj.2.weight" in image_proj_dict:  # Multi-layer projector
                weight_shape = image_proj_dict["proj.2.weight"].shape
                if weight_shape[0] % norm_size == 0:
                    self.num_tokens = weight_shape[0] // norm_size
                    logging.info(f"Detected num_tokens={self.num_tokens} from proj.2.weight shape={weight_shape}, norm_size={norm_size}")
        elif is_xlab and "ip_adapter_proj_model.proj.weight" in all_keys:  # XLabs single-layer projector
            weight_shape = state_dict["ip_adapter_proj_model.proj.weight"].shape[0]
            self.num_tokens = weight_shape // self.joint_attention_dim  # 65536 / 4096 = 16
            logging.info(f"XLabs: Detected num_tokens={self.num_tokens} from ip_adapter_proj_model.proj.weight shape={weight_shape}, joint_attention_dim={self.joint_attention_dim}")
        else:
            logging.warning("Auto-detection failed, it looks suspicious, using previous value...")

        logging.info(f"Final num_tokens={self.num_tokens}")
		
        # Handle XLabs projector
        if is_xlab:
            xlabs_proj_dict = {}
            for k in all_keys:
                if k.startswith("ip_adapter_proj_model."):
                    # Keep original key names without renaming
                    new_key = k.replace("ip_adapter_proj_model.", "")
                    xlabs_proj_dict[new_key] = state_dict[k]
                    
            if xlabs_proj_dict:
                logging.debug(f"Found XLabs proj keys: {list(xlabs_proj_dict.keys())}")  
             
                # Use projection_dim for id_embeddings_dim
                id_embeddings_dim = getattr(self.image_encoder.config, "projection_dim", 768)
                if "proj.weight" in xlabs_proj_dict:  # Check original key name
                    expected_dim = xlabs_proj_dict["proj.weight"].shape[1]  # Input size
                    if id_embeddings_dim != expected_dim:
                        logging.warning(f"XLabs: Mismatch in hidden_size: expected {expected_dim}, got {id_embeddings_dim} from projection_dim. Will use projection layer in get_image_embeds.")
                        id_embeddings_dim = expected_dim
                
                self.image_proj_model = MLPProjModel(
                    cross_attention_dim=self.joint_attention_dim,  # 4096
                    id_embeddings_dim=id_embeddings_dim,  # 768 for clip-vit-large-patch14
                    num_tokens=self.num_tokens,  # 128
                    single_layer=True
                ).to(self.device, dtype=self.dtype)
                try:
                    # Rename keys to match single-layer model
                    adjusted_dict = {}
                    for k, v in xlabs_proj_dict.items():
                        if k == "proj.weight":
                            adjusted_dict["proj.0.weight"] = v  # Map to single-layer weight
                        elif k == "proj.bias":
                            adjusted_dict["proj.0.bias"] = v  # Map to single-layer bias
                        else:
                            adjusted_dict[k] = v  # norm.weight, norm.bias unchanged
                    # Log expected keys for debugging
                    if DEBUG_ENABLED:
                        expected_keys = list(self.image_proj_model.state_dict().keys())
                        logging.debug(f"Expected keys in MLPProjModel: {expected_keys}")
                        # Check for missing keys
                        missing_keys = set(expected_keys) - set(adjusted_dict.keys())
                        if missing_keys:
                            logging.warning(f"Missing keys in xlabs_proj_dict: {missing_keys}")
                    self.image_proj_model.load_state_dict(adjusted_dict, strict=False)
                    logging.info("Loaded XLabs image_proj successfully")
                except Exception as e:
                    logging.error(f"Failed to load XLabs image_proj parameters: {e}")
                    self.image_proj_model = None  # Fallback to raw embeds
            else:
                logging.warning("No XLabs proj keys found, using raw clip_image_embeds")
                self.image_proj_model = None
        else:
            # Non-XLabs (e.g., InstantX)
            id_embeddings_dim = getattr(self.image_encoder.config, "hidden_size", 1152)  # 1152 for SigLIP
            
            image_proj_dict = state_dict.get("image_proj", {k.replace("image_proj.", ""): v for k in all_keys if k.startswith("image_proj.")})
            if image_proj_dict:
                logging.debug(f"Found image_proj keys: {list(image_proj_dict.keys())[:5]}")
                if "proj.0.weight" in image_proj_dict:
                    expected_dim = image_proj_dict["proj.0.weight"].shape[1]  # Input size of the first linear layer
                    if id_embeddings_dim != expected_dim:
                        logging.warning(f"Non-XLabs: Mismatch in hidden_size: expected {expected_dim}, got {id_embeddings_dim} from hidden_size. Will use projection layer in get_image_embeds.")
                        id_embeddings_dim = expected_dim
                
                self.image_proj_model = MLPProjModel(
                    cross_attention_dim=self.joint_attention_dim,
                    id_embeddings_dim=id_embeddings_dim, #1152
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.dtype)
                try:
                    self.image_proj_model.load_state_dict(image_proj_dict, strict=False)
                    logging.info("Loaded image_proj successfully")
                except Exception as e:
                    logging.error(f"Failed to load image_proj: {e}")
                    raise
            else:
                logging.warning("No image_proj keys found. Using raw clip_image_embeds.")
                self.image_proj_model = None

        # Parsing ip_adapter
        ip_adapter_dict = {}
        if "ip_adapter" in state_dict:
            ip_adapter_dict = state_dict["ip_adapter"]
            logging.info("Using InstantX ip_adapter format")
        else:
            for k in all_keys:
                if k.startswith("double_blocks."):
                    # Remove double_blocks prefix, keep block index
                    new_key = k.replace("double_blocks.", "")
                    new_key = new_key.replace("processor.ip_adapter_double_stream_", "")
                    # Rename k_proj and v_proj weights to match IPAFluxAttnProcessor2_0 expected names
                    if "k_proj.weight" in new_key:
                        new_key = new_key.replace("k_proj.weight", "to_k_ip.weight")
                    elif "v_proj.weight" in new_key:
                        new_key = new_key.replace("v_proj.weight", "to_v_ip.weight")
                    # Skip bias keys as IPAFluxAttnProcessor2_0 does not expect them
                    if "bias" not in new_key:
                        ip_adapter_dict[new_key] = state_dict[k]
            logging.info("Using XLabs ip_adapter format")

        if not ip_adapter_dict:
            logging.error("No ip_adapter keys found")
            raise KeyError(f"No ip_adapter keys found in {path}")

        # Initialize ipadapter
        self.ip_attn_procs = self.init_ip_adapter()
        if shuffle_weights:
            if not is_xlab:
                # For InstantX Flux, use values directly without sorting
                ip_layers = torch.nn.ModuleList(self.ip_attn_procs.values())
            else:
                # Shuffle keys randomly for non-InstantX Flux adapters
                import random
                keys = list(self.ip_attn_procs.keys())
                random.shuffle(keys)
                ip_layers = torch.nn.ModuleList([self.ip_attn_procs[key] for key in keys])
            logging.info("Using shuffled weights for IP-Adapter")
        else:
            # Use standard sorting for other adapters (Flux.1, InstantX)
            sorted_keys = sorted(self.ip_attn_procs.keys())
            ip_layers = torch.nn.ModuleList([self.ip_attn_procs[key] for key in sorted_keys])
            logging.info("Using sorted weights for IP-Adapter")
        try:
            if DEBUG_ENABLED:
                # Log expected keys for debugging
                sample_proc = list(self.ip_attn_procs.values())[0]
                logging.debug(f"Expected parameter names in IPAFluxAttnProcessor2_0: {list(sample_proc.state_dict().keys())}")
            ip_layers.load_state_dict(ip_adapter_dict, strict=False)
            logging.info("Loaded ip_adapter successfully")
        except Exception as e:
            logging.error(f"Failed to load ip_adapter: {e}")
            raise

        del state_dict

    def init_ip_adapter(self, weight=1.0, timestep_percent_range=(0.0, 1.0)):
        # Initialize attention processors for IP-Adapter
        ip_attn_procs = {}
        if hasattr(self, 'is_xlab') and self.is_xlab:
            # For XLabs models (e.g., flux-ip-adapter-v2), use only 19 double blocks
            dsb_count = 19  # Number of double blocks in flux-ip-adapter-v2
            for i in range(dsb_count):
                name = f"double_blocks.{i}"
                ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale=weight,
                    timestep_range=timestep_percent_range
                ).to(self.device, dtype=self.dtype)
            # No single blocks in XLabs flux-ip-adapter-v2
            logging.info("Initialized XLabs IP-Adapter with 19 double blocks")
        else:
            # For non-XLabs models, use default configuration
            dsb_count = 19 #len(flux_model.diffusion_model.double_blocks)
            for i in range(dsb_count):
                name = f"double_blocks.{i}"
                ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale=weight,
                    timestep_range=timestep_percent_range
                ).to(self.device, dtype=self.dtype)
            ssb_count = 38 #len(flux_model.diffusion_model.single_blocks)
            for i in range(ssb_count):
                name = f"single_blocks.{i}"
                ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale=weight,
                    timestep_range=timestep_percent_range
                ).to(self.device, dtype=self.dtype)
            logging.info("Initialized non-XLabs IP-Adapter with 19 double blocks and 38 single blocks")
        return ip_attn_procs

    def update_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        s = flux_model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))
        for ip_layer in self.ip_attn_procs.values():
            ip_layer.scale = weight
            ip_layer.timestep_range = timestep_range

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        # Generate image embeddings from PIL image or provided embeddings
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                logging.info("pil_image is a single Image, wrapping in list")
                pil_image = [pil_image]
            try:
                clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                logging.debug(f"clip_image shape: {clip_image.shape}")
            except Exception as e:
                logging.error(f"Failed to process PIL image: {e}")
                raise
            try:
                clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.dtype)).pooler_output
                clip_image_embeds = clip_image_embeds.to(dtype=self.dtype)
                logging.debug(f"clip_image_embeds shape: {clip_image_embeds.shape}, dtype: {clip_image_embeds.dtype}")
            except Exception as e:
                logging.error(f"Failed to encode image with CLIP: {e}")
                raise
        else:
            if clip_image_embeds is None:
                logging.error("No PIL image or clip_image_embeds provided")
                raise ValueError("Either pil_image or clip_image_embeds must be provided")
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=self.dtype)
            logging.debug(f"Provided clip_image_embeds shape: {clip_image_embeds.shape}, dtype: {clip_image_embeds.dtype}")

        if self.image_proj_model is None:
            logging.info("image_proj_model is None, using raw clip_image_embeds")
            target_size = self.joint_attention_dim
            if clip_image_embeds.shape[1] != target_size:
                if self.fallback_proj is None:
                    logging.info(f"Creating fallback projection: Linear({clip_image_embeds.shape[1]}, {target_size})")
                    self.fallback_proj = torch.nn.Linear(clip_image_embeds.shape[1], target_size).to(self.device, dtype=self.dtype)
                    torch.nn.init.xavier_uniform_(self.fallback_proj.weight)
                    self.fallback_proj.bias.data.fill_(0)
                clip_image_embeds = self.fallback_proj(clip_image_embeds)
                logging.debug(f"clip_image_embeds after fallback_proj: {clip_image_embeds.shape}")
            clip_image_embeds = clip_image_embeds.unsqueeze(1).repeat(1, self.num_tokens, 1)
            logging.debug(f"Final clip_image_embeds shape: {clip_image_embeds.shape}")
            return clip_image_embeds
        
        # Log image_proj_model parameters
        if hasattr(self.image_proj_model, 'id_embeddings_dim'):
            logging.debug(f"image_proj_model: id_embeddings_dim={self.image_proj_model.id_embeddings_dim}, num_tokens={self.image_proj_model.num_tokens}")
        else:
            logging.warning("image_proj_model has no id_embeddings_dim attribute")

        # Check if projection is needed
        if hasattr(self.image_proj_model, 'id_embeddings_dim'):
            expected_dim = self.image_proj_model.id_embeddings_dim
            actual_dim = clip_image_embeds.shape[-1]
            if actual_dim != expected_dim:
                logging.warning(f"Mismatch in dimensions: expected {expected_dim} for image_proj_model, got {actual_dim} from CLIP Vision. Adding projection layer.")
                # Always reinitialize embed_projection to ensure correct dimensions
                logging.info(f"Creating/Reinitializing embed_projection: Linear({actual_dim}, {expected_dim})")
                self.embed_projection = torch.nn.Linear(actual_dim, expected_dim).to(self.device, dtype=self.dtype)
                torch.nn.init.xavier_uniform_(self.embed_projection.weight)
                self.embed_projection.bias.data.fill_(0)
                logging.debug(f"embed_projection created: weight shape={self.embed_projection.weight.shape}, bias shape={self.embed_projection.bias.shape}")
                try:
                    clip_image_embeds = self.embed_projection(clip_image_embeds)
                    logging.debug(f"clip_image_embeds after embed_projection: {clip_image_embeds.shape}")
                except Exception as e:
                    logging.error(f"Failed to apply embed_projection: {e}")
                    raise
        else:
            logging.warning("image_proj_model has no id_embeddings_dim attribute, skipping projection")

        # Apply projection model
        try:
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            logging.info(f"image_prompt_embeds shape: {image_prompt_embeds.shape}, dtype={image_prompt_embeds.dtype}")
            logging.debug(f"image_prompt_embeds sample values (first 5): {image_prompt_embeds.flatten()[:5].tolist()}")
        except Exception as e:
            logging.error(f"Failed to apply image_proj_model: {e}")
            raise

        return image_prompt_embeds

class IPAdapterFluxLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter-flux"),),
                "clip_vision": (get_clip_vision_models(),),
                "provider": (["cuda", "cpu", "mps"],),
                "dtype": (["auto", "float16", "bfloat16"], {"default": "auto"}),
                "shuffle_weights": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    RETURN_NAMES = ("ipadapterFlux",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"

    def load_model(self, ipadapter, clip_vision, provider, dtype="auto", shuffle_weights="False"):
        # Load IP-Adapter model
        logging.info(f"Loading InstantX IPAdapter Flux model with clip_vision={clip_vision}, dtype={dtype}")
        if dtype == "auto":
            dtype = "float16"
            if provider == "cuda" and torch.cuda.is_bf16_supported():
                dtype = "bfloat16"
        model = InstantXFluxIPAdapterModel(
            image_encoder_path=clip_vision,
            ip_ckpt=ipadapter,
            device=provider,
            dtype=getattr(torch, dtype),
            shuffle_weights=shuffle_weights
        )
        return (model,)

class ApplyIPAdapterFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter_flux": ("IP_ADAPTER_FLUX_INSTANTX",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.75, "min": -1.0, "max": 5.0, "step": 0.05}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter_flux"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter_flux(self, model, ipadapter_flux, image, weight, start_percent, end_percent):
        pil_image = Image.fromarray((image.numpy()[0] * 255.0).astype(np.uint8))
        ipadapter_flux.update_ip_adapter(model.model, weight, (start_percent, end_percent))

		# Get image embeddings
        image_prompt_embeds = ipadapter_flux.get_image_embeds(pil_image=pil_image, clip_image_embeds=None)
        
        # Apply IP-Adapter to the model
        is_patched = is_model_patched(model.model)
        bi = model.clone()
        FluxUpdateModules(bi, ipadapter_flux.ip_attn_procs, image_prompt_embeds, is_patched)
        return (bi,)

NODE_CLASS_MAPPINGS = {
    "IPAdapterFluxLoader": IPAdapterFluxLoader,
    "ApplyIPAdapterFlux": ApplyIPAdapterFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterFluxLoader": "Load IPAdapter Flux Model",
    "ApplyIPAdapterFlux": "Apply IPAdapter Flux Model",
}