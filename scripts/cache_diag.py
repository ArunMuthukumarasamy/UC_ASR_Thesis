import sqlite3
c = sqlite3.connect("cache/llm_responses.sqlite")
print("TOTAL rows:", c.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])
print()
print("By model (top 12):")
for m, n in c.execute("SELECT model, COUNT(*) FROM llm_cache GROUP BY model ORDER BY COUNT(*) DESC LIMIT 12").fetchall():
    print(f"  {n:>8}  {m}")
print()
# Groq gpt-oss rows
groq = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE '%gpt-oss%'").fetchone()[0]
groq_def = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE '%gpt-oss%' AND relevance=3.0").fetchone()[0]
groq_empty = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE '%gpt-oss%' AND TRIM(raw)=''").fetchone()[0]
print("gpt-oss (Groq) rows:", groq)
print("  relevance==3.0 (default fallback):", groq_def)
print("  raw text empty:", groq_empty)
print()
# today only
today = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE '%gpt-oss%' AND created_at LIKE '2025-%' AND created_at > (SELECT MAX(created_at) FROM llm_cache WHERE model LIKE '%gpt-oss%')").fetchone()[0]
print("latest created_at:", c.execute("SELECT MAX(created_at) FROM llm_cache").fetchone()[0])
c.close()
