"""fp64 proof that the chunk (WY) form of the KDA verify window is EXACT.

Checks, against a plain sequential recurrence over T tokens (KDA.md §1.1):
  Lemma 1  the unrolled state identity  S_t = e^{G_t}(S_0 + Σ_s e^{-G_s} k̃_s u_sᵀ)
  Lemma 2  the updates solve the unit-lower-triangular system
           (I + tril₋₁(diag(β) A)) U = diag(β)(V − W)
  outputs  O = (q̃⊙λ)·S_0 + tril₀(B)·U   and the final state in one GEMM
  Neumann  (I+N)⁻¹ = Σ_{j<T} (−N)^j exactly (N strictly lower ⇒ nilpotent)

Everything in fp64: the only "error" left is machine epsilon (~1e-15),
i.e. the chunk form is a re-association of the same arithmetic, not an
approximation. See KDA.md Appendix A for the derivation this mirrors.
"""
import torch

torch.manual_seed(3)
dt = torch.float64
B, H, T, K, V = 2, 3, 6, 32, 32

k = torch.nn.functional.normalize(torch.randn(B, H, T, K, dtype=dt), dim=-1)
q = torch.nn.functional.normalize(torch.randn(B, H, T, K, dtype=dt), dim=-1)
v = torch.randn(B, H, T, V, dtype=dt)
g = -2.0 * torch.rand(B, H, T, K, dtype=dt)          # safe gate: g in (-2, 0)
beta = torch.rand(B, H, T, dtype=dt)
S0 = torch.randn(B, H, K, V, dtype=dt)

# ---- sequential reference (way 1): the for loop --------------------------
S = S0.clone()
o_seq = torch.empty(B, H, T, V, dtype=dt)
u_seq = torch.empty(B, H, T, V, dtype=dt)
for t in range(T):
    S = S * g[..., t, :, None].exp()                          # diag decay
    w_t = torch.einsum("bhk,bhkv->bhv", k[..., t, :], S)      # k̃ᵀS
    u_t = beta[..., t, None] * (v[..., t, :] - w_t)           # solved update
    S = S + k[..., t, :, None] * u_t[..., None, :]            # rank-1
    o_seq[..., t, :] = torch.einsum("bhk,bhkv->bhv", q[..., t, :], S)
    u_seq[..., t, :] = u_t
S_seq = S

# ---- chunk quantities ----------------------------------------------------
G = g.cumsum(dim=-2)                                          # [B,H,T,K]
lam = G.exp()                                                 # λ_t = e^{G_t}
kl = k * lam                                                  # k̃⊙λ
ki = k / lam                                                  # k̃⊘λ
ql = q * lam

# Lemma 1: S_t = e^{G_t} ⊙ (S_0 + Σ_{s<=t} e^{-G_s} k̃_s u_sᵀ)   (uses u_seq)
S_unroll = lam[..., -1, :, None] * (
    S0 + torch.einsum("bhsk,bhsv->bhkv", ki, u_seq))
print("Lemma 1 (unrolled final state):", (S_unroll - S_seq).abs().max().item())

# Lemma 2: (I + N) U = diag(β)(V − W),  N = tril₋₁(diag(β) A)
W = torch.einsum("bhtk,bhkv->bhtv", kl, S0)                   # (k̃⊙λ)·S0
A = torch.einsum("bhtk,bhsk->bhts", kl, ki)                   # A[t,s]
N = torch.tril(beta[..., None] * A, diagonal=-1)
rhs = beta[..., None] * (v - W)
resid = u_seq + torch.einsum("bhts,bhsv->bhtv", N, u_seq) - rhs
print("Lemma 2 (triangular system residual):", resid.abs().max().item())

# Exact Neumann inverse: N strictly lower-triangular => N^T = 0
eye = torch.eye(T, dtype=dt).expand(B, H, T, T)
M, P = eye.clone(), eye.clone()
for _ in range(T - 1):
    P = -N @ P
    M = M + P
print("Neumann inverse ((I+N)M - I):", ((eye + N) @ M - eye).abs().max().item())

# ---- the full chunk pipeline: no loop over T anywhere --------------------
U = torch.einsum("bhts,bhsv->bhtv", M, rhs)                   # U = M·β(V−W)
Bq = torch.tril(torch.einsum("bhtk,bhsk->bhts", ql, ki))      # tril₀(B)
O = torch.einsum("bhtk,bhkv->bhtv", ql, S0) \
    + torch.einsum("bhts,bhsv->bhtv", Bq, U)
Sf = lam[..., -1, :, None] * S0 \
    + torch.einsum("bhsk,bhsv->bhkv", k * (lam[..., -1:, :] / lam), U)
print("chunk O  vs sequential:", (O - o_seq).abs().max().item())
print("chunk U  vs sequential:", (U - u_seq).abs().max().item())
print("chunk S  vs sequential:", (Sf - S_seq).abs().max().item())
