import torch
from torch import Tensor
from .flux.layers import DoubleStreamBlockIPA, SingleStreamBlockIPA
from comfy.ldm.flux.layers import timestep_embedding
from types import MethodType
import hashlib
import logging

def FluxUpdateModules(bi, ip_attn_procs, image_emb, is_patched):
    """
    Apply IP-Adapter attention processors to the Flux model.
    
    Args:
        bi: model object to patch methods and attributes.
        ip_attn_procs: Dictionary of IP-Adapter attention processors.
        image_emb: Image embeddings for attention.
        is_patched: Boolean indicating if model is already patched.
    """
    flux_model = bi.model
    bi.add_object_patch(f"diffusion_model.forward_orig", MethodType(forward_orig_ipa, flux_model.diffusion_model))
    
    # Patch double blocks
    for i, original in enumerate(flux_model.diffusion_model.double_blocks):
        patch_name = f"double_blocks.{i}"
        if patch_name not in ip_attn_procs:
                logging.debug(f"Skipping {patch_name} as no IP-Adapter processor found")
                continue
        maybe_patched_layer = bi.get_model_object(f"diffusion_model.{patch_name}")
        # if there's already a patch there, collect its adapters and replace it        
        procs = [ip_attn_procs[patch_name]]
        embs = [image_emb]
        if isinstance(maybe_patched_layer, DoubleStreamBlockIPA):
            procs = maybe_patched_layer.ip_adapter + procs
            embs = maybe_patched_layer.image_emb + embs
        # initial ipa models with image embeddings
        new_layer = DoubleStreamBlockIPA(original, procs, embs)
        # TODO: maybe there's a different patching method that will automatically chain patches?
        # for example, ComfyUI internally uses model.add_patches to add loras
        bi.add_object_patch(f"diffusion_model.{patch_name}", new_layer)
    
    # Patch single blocks only if processors exist
    for i, original in enumerate(flux_model.diffusion_model.single_blocks):
        patch_name = f"single_blocks.{i}"
        if patch_name not in ip_attn_procs:
            logging.debug(f"Skipping {patch_name} as no IP-Adapter processor found")
            continue
        maybe_patched_layer = bi.get_model_object(f"diffusion_model.{patch_name}")
        procs = [ip_attn_procs[patch_name]]
        embs = [image_emb]
        if isinstance(maybe_patched_layer, SingleStreamBlockIPA):
            procs = maybe_patched_layer.ip_adapter + procs
            embs = maybe_patched_layer.image_emb + embs
        # initial ipa models with image embeddings
        new_layer = SingleStreamBlockIPA(original, procs, embs)
        bi.add_object_patch(f"diffusion_model.{patch_name}", new_layer)
        
def is_model_patched(model):
    def test(mod):
        if isinstance(mod, DoubleStreamBlockIPA):
            return True
        else:
            for p in mod.children():
                if test(p):
                    return True
        return False
    result = test(model)
    return result

def forward_orig_ipa(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor|None = None,
    control=None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                if isinstance(block, DoubleStreamBlockIPA): # ipadaper 
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], t=args["timesteps"], attn_mask=args.get("attn_mask"))
                else:
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))
                return out
            out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "timesteps": timesteps, "attn_mask": attn_mask}, {"original_block": block_wrap,"transformer_options": transformer_options})
            txt = out["txt"]
            img = out["img"]
        else:
            if isinstance(block, DoubleStreamBlockIPA): # ipadaper 
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, t=timesteps, attn_mask=attn_mask)
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None: # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                if isinstance(block, SingleStreamBlockIPA): # ipadaper
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], t=args["timesteps"], attn_mask=args.get("attn_mask"))
                else:
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "timesteps": timesteps, "attn_mask": attn_mask}, {"original_block": block_wrap, "transformer_options": transformer_options})
            img = out["img"]
        else:
            if isinstance(block, SingleStreamBlockIPA): # ipadaper
                img = block(img, vec=vec, pe=pe, t=timesteps, attn_mask=attn_mask)
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None: # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

    img = img[:, txt.shape[1] :, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img
    
def _compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        logging.warning(f"Failed to compute SHA256 for {file_path}: {e}")
        return None
