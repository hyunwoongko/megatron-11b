# Megatron 11B
- Porting of Megatron LM 11B model published on facebook on Huggingface Transformers.
- This repo contains the model's code, checkpoints and deepspeed parallelization examples.
<br><br>
  
## Installation
```console
pip install megatron-11b
```
<br>

## Usage
### 1. Tokenizer
- The usage of tokenizer is the same as other tokenizers of the existing Huggingface.

```python
from megatron_11b import MegatronTokenizer

tokenizer = MegatronTokenizer.from_pretrained("hyunwoongko/megatron-11B")
tokens = tokenizer.encode("Kevin is")
```
<br>

### 2. Model
- We currently support the CausalLM model and the SequenceClassification model.

```python
from megatron_11b import MegatronForCausalLM, MegatronForSequenceClassification

model_clm = MegatronForCausalLM.from_pretrained("hyunwoongko/megatron-11B")
model_clf = MegatronForSequenceClassification.from_pretrained("hyunwoongko/megatron-11B")
```
<br>

### 3. Parallelism 
- Intra-layer model parallelization can be performed using DeepSpeed's InferenceEngine.

```python
from megatron_11b import MegatronForCausalLM, MegatronTokenizer
from deepspeed import InferenceEngine

tokenizer = MegatronTokenizer.from_pretrained("hyunwoongko/megatron-11B")
model = MegatronForCausalLM.from_pretrained("hyunwoongko/megatron-11B").half()
model = InferenceEngine(model, mp_size=8, replace_method='auto').module

inputs = "Kevin is"
inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].cuda()

output = model.generate(inputs, num_beams=5, no_repeat_ngram_size=4, repetition_penalty=1.4)
print(tokenizer.batch_decode(output))
```
- And do an inference with the command below.
```console
deepspeed --num_gpus=8 inference.py
```
<br>


## References
- https://github.com/pytorch/fairseq/tree/master/examples/megatron_11b
- https://github.com/huggingface/transformers/pull/10301
- https://github.com/microsoft/DeepSpeed/pull/1168
