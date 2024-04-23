# copyright ziqi-jin
import torch
import torch.nn as nn
from .segment_anything_ori import sam_model_registry
from .image_encoder_adapter import BaseImgEncodeAdapter, LoraImgEncodeAdapter
from .mask_decoder_adapter import BaseMaskDecoderAdapter, SemMaskDecoderAdapter, LoraMaskDecoderAdapter
from .prompt_encoder_adapter import BasePromptEncodeAdapter, TextEncoderAdapter
from .utils import fix_params, fix_params_by_name


class BaseExtendSam(nn.Module):

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, model_type='vit_b'):
        super(BaseExtendSam, self).__init__()
        assert model_type in ['default', 'vit_b', 'vit_l', 'vit_h'], print(
            "Wrong model_type, SAM only can be built as vit_b, vot_l, vit_h and default ")
        self.ori_sam = sam_model_registry[model_type](ckpt_path)
        self.img_adapter = BaseImgEncodeAdapter(self.ori_sam, fix=fix_img_en)
        self.prompt_adapter = BasePromptEncodeAdapter(self.ori_sam, fix=fix_prompt_en)
        self.mask_adapter = BaseMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de)

    def forward(self, img):
        x = self.img_adapter(img)
        points = None
        boxes = None
        masks = None

        sparse_embeddings, dense_embeddings = self.prompt_adapter(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        multimask_output = True
        low_res_masks, iou_predictions = self.mask_adapter(
            image_embeddings=x,
            prompt_adapter=self.prompt_adapter,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        return low_res_masks, iou_predictions


class SemanticSam(BaseExtendSam):

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, class_num=20, model_type='vit_b'):
        super().__init__(ckpt_path=ckpt_path, fix_img_en=fix_img_en, fix_prompt_en=fix_prompt_en,
                         fix_mask_de=fix_mask_de, model_type=model_type)
        self.mask_adapter = SemMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de, class_num=class_num)


class TextSam(BaseExtendSam):

    def __init__(self, ckpt_path=None, img_en_type='fix', prompt_en_type='fix', mask_de_type='fix', model_type='vit_b', enhance_proj=False):
        
        assert (img_en_type in ['fix', 'full'] or img_en_type.startswith('lora')), img_en_type
        assert (prompt_en_type in ['fix', 'full']), prompt_en_type
        assert (mask_de_type in ['fix', 'full'] or mask_de_type.startswith('lora')), mask_de_type
        
        super().__init__(ckpt_path=ckpt_path, fix_img_en=img_en_type=='fix', fix_prompt_en=prompt_en_type=='fix',
                         fix_mask_de=mask_de_type=='fix', model_type=model_type)
        
        if (img_en_type.startswith('lora')):
            lora_rank = int(img_en_type.split('lora')[1])
            self.img_adapter = LoraImgEncodeAdapter(self.ori_sam, fix=False, rank=lora_rank)
        
        if (mask_de_type.startswith('lora')):
            lora_rank = int(mask_de_type.split('lora')[1])
            self.mask_adapter = LoraMaskDecoderAdapter(self.ori_sam, fix=False, rank=lora_rank)
        
        self.prompt_adapter = TextEncoderAdapter(self.ori_sam, fix=prompt_en_type=='fix', enhance_proj=enhance_proj)
        
        # self.mask_adapter = LoraMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de, rank=lora_rank)
        # fix_params_by_name(self.mask_adapter, ['output_upscaling'])
    
    def forward(self, img, text_array, masks=None):
        ''' text_array: len=batch size '''

        x = self.img_adapter(img)
        
        # print('got mask:', masks)
        # print('mask shape:', masks.shape)
        # print('img shape', img.shape)
        # xxx

        sparse_embeddings, dense_embeddings = self.prompt_adapter(text_array, masks=masks)


        # print('sparse', sparse_embeddings)

        multimask_output = False # only output 1 mask, for given text input
        low_res_masks, iou_predictions = self.mask_adapter(
            image_embeddings=x,
            prompt_adapter=self.prompt_adapter,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        return low_res_masks, iou_predictions


