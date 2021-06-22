# Megatron LM
- Porting of Megatron LM 11B model published on facebook on Huggingface Transformers.
- This repo contains the model's code, checkpoints and deepspeed parallelization examples.
<br><br>
  
## Installation
```console
pip install megatron-lm
```
<br>

## Usage
### 1. Tokenizer
- The usage of tokenizer is the same as other tokenizers of the existing Huggingface.
```python
from megatron_lm import MegatronTokenizer

tokenizer = MegatronTokenizer.from_pretrained("hyunwoongko/megatron-11B")
tokens = tokenizer.encode("hello. My name is Kevin.")
```
<br>

### 2. Model
- We currently support the CausalLM model and the SequenceClassification model.
```python
from megatron_lm import MegatronForCausalLM, MegatronForSequenceClassification

model_clm = MegatronForCausalLM.from_pretrained("hyunwoongko/megatron-11B")
model_clf = MegatronForSequenceClassification.from_pretrained("hyunwoongko/megatron-11B")
```
<br>

### 3. Parallelism 
- Intra-layer model parallelization can be performed using DeepSpeed's InferenceEngine.
```python
from megatron_lm import MegatronForCausalLM, MegatronTokenizer
from deepspeed import InferenceEngine
import torch.distributed as dist

tokenizer = MegatronTokenizer.from_pretrained("hyunwoongko/megatron-11B")
model = MegatronForCausalLM.from_pretrained("hyunwoongko/megatron-11B")

model = InferenceEngine(
    model=model,
    mp_size=4,
    replace_method="auto",
).module

tokens = tokenizer.encode(
    "Hello. My name is Kevin. I was", 
    return_tensors="pt",
).cuda()

output = model.generate(
    tokens, num_beams=5, max_length=40,
)

if dist.get_rank() == 0:
    print(output)
```
- And do an inference with the command below.
```console
deepspeed --num_gpus=4 inference.py
```
<br>


## References
- https://github.com/pytorch/fairseq/tree/master/examples/megatron_11b
- https://github.com/huggingface/transformers/pull/10301
- https://github.com/microsoft/DeepSpeed/pull/1168
