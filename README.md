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

### 3. Parallelism (w/ Deepspeed)
- Intra-layer model parallelization can be performed using my custom DeepSpeed's InferenceEngine.
- Deepspeed's Infernece Engine does not check whether the tensor is contiguous when broadcasting to each device. 
- When we are using the NCCL backend, tensor must be contiguous when transferring to other devices using multiprocessing.
- These changes are currently being PR and being considered by Microsoft. (https://github.com/microsoft/DeepSpeed/pull/1168)
- But since these changes haven't been implemented in the release version (0.4.0) yet, I've included it in the package.

```python
from megatron_11b import MegatronForCausalLM, MegatronTokenizer
from megatron_11b.deepspeed_custom import InferenceEngine  # Bug fixed engine

tokenizer = MegatronTokenizer.from_pretrained("hyunwoongko/megatron-11B")
model = MegatronForCausalLM.from_pretrained("hyunwoongko/megatron-11B").half()
model = InferenceEngine(model, mp_size=4, replace_method='auto').module

inputs = "Kevin is"
inputs = tokenizer.encode(inputs, return_tensors="pt").cuda()[:-1]  # exclude EOS

output = model.generate(inputs, num_beams=5, no_repeat_ngram_size=4, repetition_penalty=1.2)
print(tokenizer.batch_decode(output))
```
- And do an inference with the command below.
```console
deepspeed --num_gpus=4 inference.py
```
-
```

```

<br>


## References
- https://github.com/pytorch/fairseq/tree/master/examples/megatron_11b
- https://github.com/huggingface/transformers/pull/10301
