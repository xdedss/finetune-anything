import torch.nn as nn
from .segment_anything_ori.modeling.sam import Sam
from .segment_anything_ori.modeling.image_encoder import Attention as OriAttention
from .utils import fix_params
from . import lora_utils
import loralib

class BaseImgEncodeAdapter(nn.Module):

    def __init__(self, ori_sam: Sam, fix=False):
        super(BaseImgEncodeAdapter, self).__init__()
        self.sam_img_encoder = ori_sam.image_encoder
        if fix:
            fix_params(self.sam_img_encoder)

    def forward(self, x):
        x = self.sam_img_encoder(x)
        return x


class LoraImgEncodeAdapter(BaseImgEncodeAdapter):

    def __init__(self, ori_sam: Sam, fix=False, rank: int=8):
        super().__init__(ori_sam, fix=False)

        if fix:
            raise NotImplementedError('It makes no sense to fix a LoRA layer, check your config')
            # fix_params(self.sam_img_encoder)
        else:
            # apply lora and fix non-lora params
            def replace_fn(module: OriAttention):
                replacement = lora_utils.AttentionLora(
                    module.dim,
                    module.num_heads,
                    module.qkv_bias,
                    module.use_rel_pos,
                    module.rel_pos_zero_init,
                    module.input_size,
                    rank
                ).to(module.qkv.weight.device)
                replacement.load_state_dict(module.state_dict(), strict=False)
                return replacement
            lora_utils.replace_module_by_type(self.sam_img_encoder, OriAttention, replace_fn)
            loralib.mark_only_lora_as_trainable(self.sam_img_encoder)
