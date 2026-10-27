"""A minimal Llama forward pass (RMSNorm, rotary positions, grouped-query attention, SwiGLU) straight from safetensors,
for running models as witnesses without the transformers stack. Returns next-token logits for a prompt."""
import json, os, torch
from safetensors.torch import load_file
class Llama:
    def __init__(self, d):
        c = json.load(open(os.path.join(d, "config.json"))); self.c = c
        f = [x for x in os.listdir(d) if x.endswith(".safetensors")]
        self.w = {}
        for x in f: self.w.update({k: v.float() for k, v in load_file(os.path.join(d, x)).items()})
        self.H, self.KV, self.L = c["num_attention_heads"], c.get("num_key_value_heads", c["num_attention_heads"]), c["num_hidden_layers"]
        self.D = c["hidden_size"]; self.hd = self.D // self.H; self.eps = c.get("rms_norm_eps", 1e-5); self.theta = c.get("rope_theta", 10000.0)
    def norm(self, x, g): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * g
    def rope(self, x, n):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2).float() / self.hd)); t = torch.arange(n).float()[:, None] * inv[None]
        cos, sin = torch.cat([t.cos(), t.cos()], -1), torch.cat([t.sin(), t.sin()], -1)
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin
    @torch.no_grad()
    def logits(self, ids):
        w = self.w; n = len(ids); x = w["model.embed_tokens.weight"][torch.tensor(ids)]
        mask = torch.full((n, n), float("-inf")).triu(1)
        for l in range(self.L):
            p = f"model.layers.{l}."; h = self.norm(x, w[p + "input_layernorm.weight"])
            q = (h @ w[p + "self_attn.q_proj.weight"].T).view(n, self.H, self.hd).transpose(0, 1)
            k = (h @ w[p + "self_attn.k_proj.weight"].T).view(n, self.KV, self.hd).transpose(0, 1)
            v = (h @ w[p + "self_attn.v_proj.weight"].T).view(n, self.KV, self.hd).transpose(0, 1)
            q, k = self.rope(q, n), self.rope(k, n); rep = self.H // self.KV
            k, v = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
            a = torch.softmax(q @ k.transpose(1, 2) / self.hd ** 0.5 + mask, -1) @ v
            x = x + a.transpose(0, 1).reshape(n, self.D) @ w[p + "self_attn.o_proj.weight"].T
            h = self.norm(x, w[p + "post_attention_layernorm.weight"])
            x = x + (torch.nn.functional.silu(h @ w[p + "mlp.gate_proj.weight"].T) * (h @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
        return self.norm(x[-1], w["model.norm.weight"]) @ w["lm_head.weight"].T
