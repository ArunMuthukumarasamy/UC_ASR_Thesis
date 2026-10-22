import sqlite3
c = sqlite3.connect("cache/llm_responses.sqlite")
tot = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE '%gpt-oss%'").fetchone()[0]
dflt = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE '%gpt-oss%' AND relevance=3.0").fetchone()[0]
empty = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE '%gpt-oss%' AND TRIM(raw)=''").fetchone()[0]
print("Groq total:", tot)
print("Groq relevance==3.0 (fallback):", dflt)
print("Groq empty raw:", empty)
print("Groq usable (real parsed, not 3.0):", tot - dflt)
c.close()
