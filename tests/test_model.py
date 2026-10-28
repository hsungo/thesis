"""LayerSkipLlama must reproduce HF transformers' Llama exactly (fp32)."""

import pytest
import torch

from conftest import LLAMA3_ROPE, make_tiny_checkpoint


@pytest.mark.parametrize("variant", ["default", "tied", "llama3_rope"])
def test_matches_hf(tmp_path, variant):
    from transformers import LlamaForCausalLM

    from common.model import LayerSkipLlama

    path = make_tiny_checkpoint(tmp_path / variant, tie=variant == "tied",
                                rope=LLAMA3_ROPE if variant == "llama3_rope" else None, seed=1)
    ours = LayerSkipLlama.from_pretrained(path, device="cpu", dtype=torch.float32, max_seq=512)
    ours.allocate_cache(2, 256)
    hf = LlamaForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    # long enough to cross the llama3 low/high frequency bands (original context 64)
    ids = torch.randint(3, 64, (1, 150), generator=torch.Generator().manual_seed(0))
    ids[0, 0] = 1
    with torch.inference_mode():
        out = hf(ids, output_hidden_states=True)
        logits, hidden = ours.prefill(ids[0].tolist(), slot=1, capture_exits=True)
        assert torch.allclose(logits, out.logits[0, -1], atol=1e-4)
        for k in range(1, ours.n_layers):  # exit-layer hidden states (HF hidden_states[k] = after layer k-1)
            assert torch.allclose(hidden[k - 1], out.hidden_states[k][0, -1], atol=1e-4)

        # one cached decode step == recomputing the whole sequence
        nxt = int(logits.argmax())
        pos = torch.tensor([[ids.shape[1]]])
        ctx = ours.make_ctx(pos, b0=1, max_pos=ids.shape[1])
        step = ours.head(ours.forward_layers(ours.embed(torch.tensor([[nxt]])), 0, ours.n_layers, ctx)[:, -1])[0]
        ref = hf(torch.cat([ids, torch.tensor([[nxt]])], 1)).logits[0, -1]
        assert torch.allclose(step, ref, atol=1e-4)


def test_config_parsing_old_and_new_formats(tmp_path):
    import json

    from common.model import ModelConfig

    d = tmp_path / "old"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "vocab_size": 10, "hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
        "intermediate_size": 16, "rope_theta": 500000.0,
        "rope_scaling": {"rope_type": "llama3", "factor": 32.0}, "eos_token_id": [5, 6]}))
    c = ModelConfig.from_hf_dir(d)
    assert c.rope_theta == 500000.0 and c.rope_scaling["factor"] == 32.0 and c.eos_token_ids == (5, 6)
    d2 = tmp_path / "new"
    d2.mkdir()
    (d2 / "config.json").write_text(json.dumps({
        "vocab_size": 10, "hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
        "intermediate_size": 16, "rope_parameters": {"rope_type": "default", "rope_theta": 1234.0}}))
    c2 = ModelConfig.from_hf_dir(d2)
    assert c2.rope_theta == 1234.0 and c2.rope_scaling is None


def test_batched_rows_are_independent(tiny_model):
    """A request's logits must not depend on the other rows of the batch (masking)."""
    m = tiny_model
    m.allocate_cache(3, 128)
    seqs = [[1, 5, 9, 11], [1, 7, 3, 3, 8, 20, 4], [1, 40]]
    with torch.inference_mode():
        last = []
        for i, s in enumerate(seqs):
            logits, _ = m.prefill(s, slot=i)
            last.append(int(logits.argmax()))
        pos = torch.tensor([[len(s)] for s in seqs])
        ctx = m.make_ctx(pos, 0, int(pos.max()))
        batched = m.head(m.forward_layers(m.embed(torch.tensor(last).unsqueeze(1)), 0, m.n_layers, ctx)[:, -1])
        for i, s in enumerate(seqs):
            m.allocate_cache(1, 128)
            m.prefill(s, slot=0)
            ctx1 = m.make_ctx(torch.tensor([[len(s)]]), 0, len(s))
            single = m.head(m.forward_layers(m.embed(torch.tensor([[last[i]]])), 0, m.n_layers, ctx1)[:, -1])
            assert torch.allclose(batched[i], single[0], atol=1e-4)
            m.allocate_cache(3, 128)
            for j, s2 in enumerate(seqs):
                m.prefill(s2, slot=j)
