# Rough but honest LLM-call budget, accounting for the SQLite cache sharing
# the N=3 sampling across UARO/ATM/UC-ASR (same prompt,temp,seed -> computed once).
K = 50          # candidates per user
N = 3           # stochastic passes (UARO/ATM/UC-ASR share these via cache)
DEEP = 2        # UC-ASR unstable deep samples (only some users/items)
SEEDS = 1       # groq profile uses 1 seed

def budget(users, ablations=0):
    # Unique calls per user per seed:
    singlepass = K * 1          # T=0.0, pass 0  -> distinct temp from sampling, NOT shared
    sampling   = K * N          # T=0.7 passes 0..2 -> SHARED by uaro/atm/uc-asr
    # gate extras (upper bound): assume ~30% unstable * DEEP, ~listwise negligible unique
    gate_extra = int(0.3 * K * DEEP)
    per_user = singlepass + sampling + gate_extra
    main = per_user * users * SEEDS
    # ablations re-score with tweaked FinalScore but REUSE cached sampling;
    # most ablations add 0 new calls (they change weights, not prompts).
    # gate_off_fixed_n3 reuses sampling too. So ablation marginal calls ~ small.
    abl = 0
    return main, per_user

for u in [10, 15, 25, 50]:
    total, per = budget(u)
    print(f"{u:>3} users: ~{total:>6,} unique LLM calls  ({per}/user)")

print()
print("Cost on Groq Dev tier (gpt-oss-20b, ~300 tok/call in+out, ~$0.10-0.30 / 1M tok est):")
for u in [15, 50]:
    total, per = budget(u)
    toks = total * 300
    lo = toks/1e6 * 0.10
    hi = toks/1e6 * 0.50
    print(f"  {u} users: ~{total:,} calls x ~300 tok = ~{toks/1e6:.2f}M tok -> ~${lo:.2f}-${hi:.2f}")

print()
print("Free-tier days to finish (~500 usable real calls/day):")
for u in [10, 15, 50]:
    total, per = budget(u)
    print(f"  {u} users: ~{total:,} calls -> ~{total/500:.0f} days of free-tier grind")
