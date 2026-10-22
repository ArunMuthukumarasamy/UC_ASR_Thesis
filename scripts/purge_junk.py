import sqlite3
c = sqlite3.connect("cache/llm_responses.sqlite")
before = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE '%gpt-oss%'").fetchone()[0]
junk = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE '%gpt-oss%' AND (relevance=3.0 OR TRIM(raw)='')").fetchone()[0]
c.execute("DELETE FROM llm_cache WHERE model LIKE '%gpt-oss%' AND (relevance=3.0 OR TRIM(raw)='')")
c.commit()
after = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE '%gpt-oss%'").fetchone()[0]
print(f"Groq rows before: {before}, junk removed: {junk}, after: {after}")
c.execute("VACUUM")
c.close()
print("purge + vacuum done")
