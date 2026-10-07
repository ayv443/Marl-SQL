# generation with vllm (fast) or plain huggingface (slower)
from common import MAX_COMPLETION_LEN, MODEL_NAME


class Generator:
    def __init__(self, adapter=None, engine="vllm", base=MODEL_NAME):
        self.engine = engine
        self.lora = None
        if engine == "vllm":
            from vllm import LLM
            from vllm.lora.request import LoRARequest
            self.llm = LLM(model=base, dtype="half", gpu_memory_utilization=0.85, max_model_len=8192,
                           enable_prefix_caching=True, seed=0, enable_lora=adapter is not None, max_lora_rank=64)
            if adapter:
                self.lora = LoRARequest("adapter", 1, adapter)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(base)
            self.tok.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cuda")
            if adapter:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model.eval()

    def generate(self, prompts, n=1, temperature=0.0, batch_size=8):
        # returns n outputs for each prompt
        if self.engine == "vllm":
            from vllm import SamplingParams
            sp = SamplingParams(n=n, temperature=temperature, max_tokens=MAX_COMPLETION_LEN)
            outs = self.llm.chat(prompts, sp, use_tqdm=True, lora_request=self.lora)
            return [[o.text for o in out.outputs] for out in outs]
        return self._hf_generate(prompts, n, temperature, batch_size)

    def _hf_generate(self, prompts, n, temperature, batch_size):
        import torch
        from tqdm import tqdm

        texts = [self.tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        results = [None] * len(texts)
        for s in tqdm(range(0, len(order), batch_size), desc="generating"):
            idx = order[s:s + batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=MAX_COMPLETION_LEN, num_return_sequences=n,
                    do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                    top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
            decoded = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, i in enumerate(idx):
                results[i] = decoded[j * n:(j + 1) * n]
        return results
