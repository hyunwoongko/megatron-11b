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
- BOS and EOS token are automatically attached, so if you want to use it as a prompt, please exclude EOS token (using `[:-1]`)
```python
from megatron_11b import MegatronTokenizer

tokenizer = MegatronTokenizer.from_pretrained("hyunwoongko/megatron-11B")
tokens = tokenizer.encode("Kevin is")
# [0, 21910, 16] ---> include EOS
tokens = tokenizer.encode("Kevin is")[:, :-1]
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
model = MegatronForCausalLM.from_pretrained("hyunwoongko/megatron-11B").half()

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

## References
- https://github.com/pytorch/fairseq/tree/master/examples/megatron_11b
- https://github.com/huggingface/transformers/pull/10301
