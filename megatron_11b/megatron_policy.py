from parallelformers.policies.base import Policy, Layer
from parallelformers.utils.dist_utils import AllReduceLinear
from megatron_11b.modeling_megatron import MegatronDecoderLayer


class MegatronPolicy(Policy):

    @staticmethod
    def replace_arguments(config, world_size):
        return {
            # 1. reduce hidden size
            "self_attn.embed_dim": config.d_model // world_size,

            # 2. reduce number of heads
            "self_attn.num_heads": config.encoder_attention_heads // world_size,
        }

    @staticmethod
    def attn_qkv():
        return [
            Layer(
                weight="self_attn.q_proj.weight",
                bias="self_attn.q_proj.bias",
            ),
            Layer(
                weight="self_attn.k_proj.weight",
                bias="self_attn.k_proj.bias",
            ),
            Layer(
                weight="self_attn.v_proj.weight",
                bias="self_attn.v_proj.bias",
            ),
        ]

    @staticmethod
    def attn_out():
        return [
            Layer(
                weight="self_attn.out_proj.weight",
                bias="self_attn.out_proj.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Layer(
                weight="fc1.weight",
                bias="fc1.bias",
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="fc2.weight",
                bias="fc2.bias",
                replace=AllReduceLinear,
            ),
        ]

    @staticmethod
    def original_layer_class():
        return MegatronDecoderLayer
