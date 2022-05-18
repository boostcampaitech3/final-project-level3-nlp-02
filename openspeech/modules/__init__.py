# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .wrapper import Linear, View, Transpose
from .additive_attention import AdditiveAttention
from .add_normalization import AddNorm
from .batchnorm_relu_rnn import BNReluRNN
from .conformer_attention_module import MultiHeadedSelfAttentionModule
from .conformer_block import ConformerBlock
from .conformer_convolution_module import ConformerConvModule
from .conformer_feed_forward_module import FeedForwardModule
from .conv2d_extractor import Conv2dExtractor
from .conv2d_subsampling import Conv2dSubsampling
from .deepspeech2_extractor import DeepSpeech2Extractor
from .vgg_extractor import VGGExtractor
from .conv_base import BaseConv1d
from .conv_group_shuffle import ConvGroupShuffle
from .depthwise_conv1d import DepthwiseConv1d
from .glu import GLU
from .jasper_subblock import JasperSubBlock
from .jasper_block import JasperBlock
from .location_aware_attention import LocationAwareAttention
from .mask import get_attn_pad_mask, get_attn_subsequent_mask
from .mask_conv1d import MaskConv1d
from .mask_conv2d import MaskConv2d
from .multi_head_attention import MultiHeadAttention
from .pointwise_conv1d import PointwiseConv1d
from .positional_encoding import PositionalEncoding
from .positionwise_feed_forward import PositionwiseFeedForward
from .quartznet_subblock import QuartzNetSubBlock
from .quartznet_block import QuartzNetBlock
from .relative_multi_head_attention import RelativeMultiHeadAttention
from .residual_connection_module import ResidualConnectionModule
from .dot_product_attention import DotProductAttention
from .swish import Swish
from .time_channel_separable_conv1d import TimeChannelSeparableConv1d
from .transformer_embedding import TransformerEmbedding


__all__ = [
    "AdditiveAttention",
    "AddNorm",
    "BNReluRNN",
    "MultiHeadAttention",
    "ConformerBlock",
    "ConformerConvModule",
    "FeedForwardModule",
    "Conv2dExtractor",
    "Conv2dSubsampling",
    "DeepSpeech2Extractor",
    "VGGExtractor",
    "BaseConv1d",
    "ConvGroupShuffle",
    "DepthwiseConv1d",
    "GLU",
    "JasperSubBlock",
    "JasperBlock",
    "LocationAwareAttention",
    "get_attn_pad_mask",
    "get_attn_subsequent_mask",
    "MaskConv1d",
    "MaskConv2d",
    "MultiHeadAttention",
    "PointwiseConv1d",
    "PositionalEncoding",
    "PositionwiseFeedForward",
    "QuartzNetSubBlock",
    "QuartzNetBlock",
    "RelativeMultiHeadAttention",
    "ResidualConnectionModule",
    "DotProductAttention",
    "Swish",
    "TimeChannelSeparableConv1d",
    "TransformerEmbedding",
    "Linear",
    "View",
    "Transpose",
]
