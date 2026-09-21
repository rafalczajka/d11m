# d11m

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
