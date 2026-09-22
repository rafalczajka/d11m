# d11m (Dumb LLM)

An LLM built for fun and learning purposes. It uses a
GPT-like Transformer architecture.

## Architecture

<div align="center">
<pre>
Inputs
↓
Token Embedding
+
Position Embedding
↓
┌─────────────────────────┐
│        LayerNorm        │
│            ↓            │
│  Causal Self-Attention  │
│            ↓            │
│        + Residual       │
│            ↓            │
│        LayerNorm        │
│            ↓            │
│    MLP / Feed Forward   │
│            ↓            │
│        + Residual       │
   └─────────────────────────┘ Nx
↓
LayerNorm
↓
Linear
↓
Softmax
</pre>
</div>
