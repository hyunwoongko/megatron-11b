from parallelformers import parallelize
from megatron_11b import MegatronForCausalLM, MegatronTokenizer
from megatron_11b.megatron_policy import MegatronPolicy

model = MegatronForCausalLM.from_pretrained("hyunwoongko/megatron-11B")
tokenizer = MegatronTokenizer.from_pretrained("hyunwoongko/megatron-11B")
policy = MegatronPolicy

parallelize(
    model,
    gpus=[0, 1],
    fp16=True,
    custom_policy_cls=policy,
)
