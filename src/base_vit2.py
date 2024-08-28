# this is a direct copy from https://github.com/lukemelas/PyTorch-Pretrained-ViT.git


import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch
from torch.utils import model_zoo
from typing import Optional
import math


def get_base_config():
    """Base ViT config ViT"""
    return dict(
        dim=768,
        ff_dim=3072,
        num_heads=12,
        num_layers=12,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representation_size=768,
        classifier='token'
    )


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(patches=(16, 16)))
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(patches=(32, 32)))
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        patches=(16, 16),
        dim=1024,
        ff_dim=4096,
        num_heads=16,
        num_layers=24,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representation_size=1024
    ))
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(patches=(32, 32)))
    return config


def drop_head_variant(config):
    config.update(dict(representation_size=None))
    return config


PRETRAINED_MODELS = {
    'B_16': {
        'config': get_b16_config(),
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth"
    },
    'B_32': {
        'config': get_b32_config(),
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32.pth"
    },
    'L_16': {
        'config': get_l16_config(),
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': None
    },
    'L_32': {
        'config': get_l32_config(),
        'num_classes': 21843,
        'image_size': (224, 224),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32.pth"
    },
    'B_16_imagenet1k': {
        'config': drop_head_variant(get_b16_config()),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth"
    },
    'B_32_imagenet1k': {
        'config': drop_head_variant(get_b32_config()),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32_imagenet1k.pth"
    },
    'L_16_imagenet1k': {
        'config': drop_head_variant(get_l16_config()),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16_imagenet1k.pth"
    },
    'L_32_imagenet1k': {
        'config': drop_head_variant(get_l32_config()),
        'num_classes': 1000,
        'image_size': (384, 384),
        'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32_imagenet1k.pth"
    },
}


def load_pretrained_weights(
    model,
    model_name=None,
    weights_path=None,
    load_first_conv=True,
    load_fc=True,
    load_repr_layer=False,
    resize_positional_embedding=False,
    verbose=True,
    strict=True,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    assert bool(model_name) ^ bool(
        weights_path), 'Expected exactly one of model_name or weights_path'

    # Load or download weights
    if weights_path is None:
        url = PRETRAINED_MODELS[model_name]['url']
        if url:
            state_dict = model_zoo.load_url(url)
        else:
            raise ValueError(
                f'Pretrained model for {model_name} has not yet been released')
    else:
        state_dict = torch.load(weights_path)

    # Modifications to load partial state dict
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight',
                                  'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    # Change size of positional embeddings
    if resize_positional_embedding:
        posemb = state_dict['positional_embedding.pos_embedding']
        posemb_new = model.state_dict()['positional_embedding.pos_embedding']
        state_dict['positional_embedding.pos_embedding'] = \
            resize_positional_embedding_(posemb=posemb, posemb_new=posemb_new,
                                         has_class_token=hasattr(model, 'class_token'))
        maybe_print('Resized positional embeddings from {} to {}'.format(
                    posemb.shape, posemb_new.shape), verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        assert set(ret.missing_keys) == set(expected_missing_keys), \
            'Missing keys when loading pretrained weights: {}'.format(
                ret.missing_keys)
        assert not ret.unexpected_keys, \
            'Missing keys when loading pretrained weights: {}'.format(
                ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('Missing keys when loading pretrained weights: {}'.format(
            ret.missing_keys), verbose)
        maybe_print('Unexpected keys when loading pretrained weights: {}'.format(
            ret.unexpected_keys), verbose)
        return ret


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def resize_positional_embedding_(posemb, posemb_new, has_class_token=True):
    """Rescale the grid of position embeddings in a sensible manner"""
    from scipy.ndimage import zoom

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    # Rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)

    # Deal with class token and return
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class CustomLayerNorm(nn.Module):
    def __init__(self, dim, num_particles,eps=1e-6):
        super(CustomLayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.num_particles = num_particles
    def forward(self, x):
        final_res = []
        for i in range(self.num_particles):
            mean = x[i].mean(-1, keepdim=True)
            std = x[i].std(-1, keepdim=True)
            res = self.weight * (x[i] - mean) / (std + self.eps) + self.bias
            final_res.append(res)
        return final_res
    
class CustomLinear(nn.Module):
    def __init__(self, dim_in, dim_out, num_particles):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim_in))
        self.bias = nn.Parameter(torch.zeros(dim_out))
        self.num_particles = num_particles
        
    def forward(self, x):
        final_res = []
        for i in range(self.num_particles):
            res = torch.nn.functional.linear(x[i], self.weight, self.bias) #torch.matmul(x[i], self.weight) + self.bias
            final_res.append(res)
            
        return final_res
    
        
class CustomLinear2(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_particles=1):
        super(CustomLinear2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.num_particles = num_particles
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.sqrt(torch.tensor(fan_in, dtype=torch.float))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        
        final_res = []
        for i in range(self.num_particles):
            res = torch.nn.functional.linear(x[i], self.weight, self.bias) #torch.nn.functional.linear(x[i], self.weight, self.bias) #torch.matmul(x[i], self.weight) + self.bias
            final_res.append(res)
                    
        return final_res#[0]
            
class CustomDropout(nn.Module):
    def __init__(self, dropout, num_particles):
        super(CustomDropout, self).__init__()
        self.dropout = dropout
        self.num_particles = num_particles

    def forward(self, x):
        # print('do', type(x), len(x), x[0].shape)
        if not self.training:
            return x
        final_res = []
        for i in range(self.num_particles):
            mask = (torch.rand_like(x[i].clone().detach()) > self.dropout).float()
            res = mask * x[i] / (1 - self.dropout)
            final_res.append(res)
        return final_res
    
class SimpleDropout(nn.Module):
    def __init__(self, dropout):
        super(SimpleDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout == 0.0:
            return x
        mask = (torch.rand_like(x.clone().detach()) > self.dropout).float()
        return mask * x / (1 - self.dropout)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""

    def __init__(self, dim, num_heads, dropout, num_particles):
        super().__init__()
        self.proj_q = CustomLinear(dim, dim, num_particles)
        self.proj_k = CustomLinear(dim, dim, num_particles)
        self.proj_v = CustomLinear(dim, dim, num_particles)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None  # for visualization
        self.num_particles = num_particles

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        h = []
        for i in range(self.num_particles):
            q_i, k_i, v_i = q[i], k[i], v[i]
            q_i, k_i, v_i = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q_i, k_i, v_i])
            # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
            scores = q_i @ k_i.transpose(-2, -1) / np.sqrt(k_i.size(-1))
            if mask is not None:
                mask = mask[:, None, None, :].float()
                scores -= 10000.0 * (1.0 - mask)
   
            scores = F.softmax(scores, dim=-1)
            # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
            h_i = (scores @ v_i).transpose(1, 2).contiguous()
            h_i = merge_last(h_i, 2)
            
            h.append(h_i)
            
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim, num_particles):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.num_particles = num_particles

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        res = []
        for i in range(self.num_particles):
            res_i = self.fc2(F.gelu(self.fc1(x[i])))
            res.append(res_i)
        return res
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, ff_dim, dropout, num_particles):
        super().__init__()
        
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout, num_particles)
        self.proj = CustomLinear(dim, dim, num_particles).cuda()
        self.norm1 = CustomLayerNorm(dim, num_particles, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim, num_particles)
        self.norm2 = CustomLayerNorm(dim, num_particles, eps=1e-6)
        self.drop = CustomDropout(dropout, num_particles)
        self.num_particles = num_particles

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))[0]
        x = x[0] + h
        x = [x]
        h = self.drop(self.pwff(self.norm2(x)))[0]
        x = x[0] + h
        return [x]
        

class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, num_particles):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout, num_particles) for _ in range(num_layers)])
        self.num_particles = num_particles

    def forward(self, x, mask=None):
        orginal_x =  x[0]
        x = []
        for i in range(self.num_particles):
            x.append(orginal_x.detach())
            
        for block in self.blocks:
            x = block(x, mask)
            
        return x


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000
    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self,
        name: Optional[str] = None,
        pretrained: bool = False,
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3,
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        num_particles: int = 1,
    ):
        super().__init__()

        # Configuration
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1

        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()

        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads,
                                       ff_dim=ff_dim, dropout=dropout_rate, num_particles = num_particles)

        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
            # print('vooo')
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = CustomLayerNorm(pre_logits_size, num_particles, eps=1e-6)
        self.fc = CustomLinear(pre_logits_size, num_classes, num_particles)
        self.num_particles = num_particles

        # Initialize weights
        self.init_weights()

        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self, None,
                weights_path='checkpoint/B_16_imagenet1k.pth',
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(
                    image_size != pretrained_image_size),
            )

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    # nn.init.constant(m.bias, 0)
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.
        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(
                b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1,d
        x = [x]
        x = self.transformer(x)  # b,gh*gw+1,d
        print('out transs')
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
            print('pre_logit', x.shape)
        if hasattr(self, 'fc'):
            x = self.norm(x)# [:, 0]  # b,d
            res = []
            for i in range(self.num_particles):
                res_i = x[i][:, 0]  # b,d
                res.append(res_i)
            x = self.fc(res)  # b,num_classes

        return x