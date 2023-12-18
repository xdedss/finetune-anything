# copyright ziqi-jin

import torch
import torch.nn as nn
from .segment_anything_ori.modeling.sam import Sam
from .utils import fix_params
import clip

import einops

class BasePromptEncodeAdapter(nn.Module):

    def __init__(self, ori_sam: Sam, fix=False):
        super(BasePromptEncodeAdapter, self).__init__()

        self.sam_prompt_encoder = ori_sam.prompt_encoder
        if fix:
            fix_params(self.sam_prompt_encoder)

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
        return sparse_embeddings, dense_embeddings

class TextEncoderAdapter(BasePromptEncodeAdapter):

    def __init__(self, ori_sam: Sam, fix=False):
        super().__init__(ori_sam, fix=True) # the original model is a dummy so it's always fixed

        model, preprocess = clip.load("ViT-B/32", device=ori_sam.device)
        self.clip = model

        self.projection = nn.Sequential(
            nn.Linear(512, 256),
        )

        if fix:
            fix_params(self.clip) # the real meat here is CLIP

    def forward(self, text_array):

        # dummy_box = torch.randn((1, 1, 4)).to(self.sam_prompt_encoder._get_device())
        dummy_sparse_emb, dummy_dense_emb = self.sam_prompt_encoder(None, None, None)

        text_emb = clip.tokenize(text_array).to(self.sam_prompt_encoder._get_device())
        text_features = self.clip.encode_text(text_emb) # bs, 512
        text_proj = self.projection(text_features) # bs, 256

        # print('shapes (sparse, dense, text):', dummy_sparse_emb.shape, dummy_dense_emb.shape, text_features.shape)
        # x

        sparse_emb = torch.cat([
            einops.repeat(dummy_sparse_emb, '1 n d -> b n d', b=text_proj.shape[0]), 
            einops.rearrange(text_proj, 'bs dim -> bs 1 dim')
            ], dim=1)
        # sparse_emb = dummy_sparse_emb

        # print('sparse after cat:', sparse_emb.shape)

        return sparse_emb, dummy_dense_emb
        