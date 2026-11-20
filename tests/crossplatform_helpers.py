"""Tiny, local random models. No tests read the user's model library."""

from types import SimpleNamespace


def tiny_model(path, family="llama"):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    torch.manual_seed(17)
    vocab = ["<unk>", "<s>", "</s>"] + sorted(ByteLevel.alphabet())
    tokenizer = Tokenizer(
        BPE(
            dict(zip(vocab, range(len(vocab)), strict=True)),
            merges=[],
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    args = dict(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    model = (
        LlamaForCausalLM(LlamaConfig(**args))
        if family == "llama"
        else Qwen2ForCausalLM(Qwen2Config(**args))
    )
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def torch_config(path, device="cpu"):
    from songe.config import Config

    config = Config(
        model=str(path),
        rank=2,
        lora_scale=2.0,  # alpha / rank, as in the default 128 / 64 (DESIGN 3.4)
        response_tokens=3,
        max_sequence=512,
        hindsight_tokens=16,
        token_chunk=2,
        accumulation=1,
        learning_rate=0.01,
    )
    config.torch_options = {"device": device, "force_cpu": True}
    return config


def fake_platform(root, name="linux"):
    return SimpleNamespace(
        name=name,
        data_dir=lambda: root,
        memory=lambda: SimpleNamespace(total=2**41, available=2**40, pressure="normal"),
        accelerators=lambda: [],
    )


def write_gguf(path, *, architecture="llama", file_type=15):
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(path, architecture)
    writer.add_file_type(file_type)
    writer.add_tensor("test.weight", np.ones((32, 32), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path
