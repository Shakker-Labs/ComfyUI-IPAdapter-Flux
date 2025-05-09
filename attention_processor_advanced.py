import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from diffusers.models.normalization import RMSNorm
from einops import rearrange

class IPAFluxAttnProcessor2_0Advanced(nn.Module):
    _instances = weakref.WeakSet()
    _global_call_count = 0
    _last_timestep_printed = None
    _first_instance_for_timestep = None  # Add this line
    
    def __init__(self, num_tokens, hidden_size, cross_attention_dim=None, scale_start=1.0, scale_end=1.0, total_steps=1, timestep_range=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale_start = scale_start
        self.scale_end = scale_end
        self.total_steps = total_steps
        self.num_tokens = num_tokens
        
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)
        self.norm_added_v = RMSNorm(128, eps=1e-5, elementwise_affine=False)
        self.timestep_range = timestep_range

        self.seen_timesteps = set()
        self.steps = 0
        
        # Add this instance to the set of instances
        self.__class__._instances.add(self)
    
    def clear_memory(self):
        self.seen_timesteps.clear()
        if hasattr(self, 'to_k_ip'):
            del self.to_k_ip
        if hasattr(self, 'to_v_ip'):
            del self.to_v_ip
        if hasattr(self, 'norm_added_k'):
            del self.norm_added_k
        if hasattr(self, 'norm_added_v'):
            del self.norm_added_v

    @classmethod
    def reset_all_instances(cls):
        """Reset all instances of the class"""
        cls._global_call_count = 0
        cls._last_timestep_printed = None
        cls._first_instance_for_timestep = None  # Add this line
        for instance in cls._instances:
            instance.seen_timesteps.clear()
            instance.steps = 0

    def reset_steps(self):
        """Reset the steps counter and seen timesteps for this instance."""
        self.seen_timesteps.clear()
        self.steps = 0
        self.__class__._last_timestep_printed = None
        # print(f"Steps and seen timesteps have been reset for this instance.")

    def __del__(self):
        self.clear_memory()
        # Remove this instance from the set when it's deleted
        self.__class__._instances.remove(self)
            
    def __call__(
        self,
        num_heads,
        query,
        image_emb: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        current_timestep = t[0].item()
        
        # Reset steps when starting a new sequence (timestep = 1.0)
        if abs(current_timestep - 1.0) < 1e-6:
            self.reset_steps()
            
        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                return None
        
        # Only update steps and print for the first instance that sees this timestep
        if current_timestep not in self.seen_timesteps:
            self.seen_timesteps.add(current_timestep)
            self.steps += 1
            
            # Only print if this is the first instance for this timestep
            if self.__class__._first_instance_for_timestep is None:
                self.__class__._first_instance_for_timestep = self
            
            if (self.__class__._first_instance_for_timestep == self and 
                current_timestep != self.__class__._last_timestep_printed):
                current_step = min(self.steps, self.total_steps)
                if self.total_steps > 1:
                    scale = self.scale_start + (self.scale_end - self.scale_start) * (current_step - 1) / (self.total_steps - 1)
                else:
                    scale = self.scale_end
                    
                # print(f"Timestep: {current_timestep}, Step: {self.steps}/{self.total_steps}, Weight: {scale}")
                self.__class__._last_timestep_printed = current_timestep
        
        # Calculate scale for return value
        current_step = min(self.steps, self.total_steps)
        if self.total_steps > 1:
            scale = self.scale_start + (self.scale_end - self.scale_start) * (current_step - 1) / (self.total_steps - 1)
        else:
            scale = self.scale_end
            
        ip_hidden_states = image_emb
        ip_hidden_states_key_proj = self.to_k_ip(ip_hidden_states)
        ip_hidden_states_value_proj = self.to_v_ip(ip_hidden_states)

        ip_hidden_states_key_proj = rearrange(ip_hidden_states_key_proj, 'B L (H D) -> B H L D', H=num_heads)
        ip_hidden_states_value_proj = rearrange(ip_hidden_states_value_proj, 'B L (H D) -> B H L D', H=num_heads)

        ip_hidden_states_key_proj = self.norm_added_k(ip_hidden_states_key_proj)
        ip_hidden_states_value_proj = self.norm_added_v(ip_hidden_states_value_proj)

        ip_hidden_states = F.scaled_dot_product_attention(query.to(image_emb.device).to(image_emb.dtype), 
                                                        ip_hidden_states_key_proj, 
                                                        ip_hidden_states_value_proj, 
                                                        dropout_p=0.0, is_causal=False)

        ip_hidden_states = rearrange(ip_hidden_states, "B H L D -> B L (H D)", H=num_heads)
        ip_hidden_states = ip_hidden_states.to(query.dtype).to(query.device)

        return scale * ip_hidden_states
