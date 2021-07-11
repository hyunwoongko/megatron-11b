# Megatron 11B
- Porting of Megatron LM 11B model published on facebook on Huggingface Transformers.
- This repo contains the model's code, checkpoints and parallelization examples.
  <br><br>

## Installation
```console
pip install megatron-11b
```
<br>

## Usage
### 1. Tokenizer
- The usage of tokenizer is the same as other tokenizers of the existing Huggingface.
- BOS and EOS token are automatically attached, so if you want to use it as a prompt, please exclude EOS token (using `[:-1]`)
```python
from megatron_11b import MegatronTokenizer

tokenizer = MegatronTokenizer.from_pretrained("hyunwoongko/megatron-11B")
tokens = tokenizer.encode("Kevin is")
# [0, 21910, 16] ---> include EOS
tokens = tokenizer.encode("Kevin is")[:-1]
# [0, 21910, 16, 2] ---> exclude EOS
```
<br>

### 2. Model
- We currently support the CausalLM model and the SequenceClassification model.
- The usage of model is also the same as other models of the existing Huggingface.

```python
from megatron_11b import MegatronForCausalLM, MegatronForSequenceClassification

model_clm = MegatronForCausalLM.from_pretrained("hyunwoongko/megatron-11B")
model_clf = MegatronForSequenceClassification.from_pretrained("hyunwoongko/megatron-11B")
```
<br>


### 3. Generation
```python
from megatron_11b import MegatronForCausalLM, MegatronTokenizer

tokenizer = MegatronTokenizer.from_pretrained("hyunwoongko/megatron-11B")
model = MegatronForCausalLM.from_pretrained("hyunwoongko/megatron-11B").half().cuda()

inputs = "Kevin is"
inputs = tokenizer.encode(inputs, return_tensors="pt").cuda()[:, :-1]  # exclude EOS

output = model.generate(inputs, num_beams=5, no_repeat_ngram_size=4, repetition_penalty=1.2)
print(tokenizer.batch_decode(output))
```
- output of generation.
```
<s>Kevin is a great guy.</s>
```
<br>

### 4. Model Parallelism
- Currently, I'm preparing an opensource called `Parallelformers` that can parallelize all models of Huggingface Transformers. 
- I plan to support model parallelization through this library. (maybe I can release it next month)
- The relevant code can be found via `MegatronPolicy` object below.
```python
from parallelformers.polices.base import Policy, Layer
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
```
<br><br>


## References
- https://github.com/pytorch/fairseq/tree/master/examples/megatron_11b
- https://github.com/huggingface/transformers/pull/10301
