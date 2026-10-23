"""Purge cached LLM rows that are the midpoint-3.0 fallback (parse failures) or
have empty raw text, so a later run RE-SCORES them under an improved parser.

Usage:
  python scripts/purge_junk.py                 # DRY RUN: show counts only
  python scripts/purge_junk.py --apply         # delete the 3.0/empty rows
  python scripts/purge_junk.py --apply --model "%Qwen%"   # restrict to a model LIKE pattern

Default model pattern is '%Qwen%' (the kaggle_unc / local HF model). Pass
--model '%' to target every model. Only rows with relevance=3.0 OR empty raw
are ever deleted; genuine parsed scores are never touched.
"""
import argparse, sqlite3, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cache/llm_responses.sqlite")
    ap.add_argument("--model", default="%Qwen%", help="SQL LIKE pattern for model id")
    ap.add_argument("--apply", action="store_true", help="actually delete (default is dry-run)")
    args = ap.parse_args()

    c = sqlite3.connect(args.db)
    total = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE ?", (args.model,)).fetchone()[0]
    junk = c.execute(
        "SELECT COUNT(*) FROM llm_cache WHERE model LIKE ? AND (relevance=3.0 OR TRIM(raw)='')",
        (args.model,),
    ).fetchone()[0]
    print(f"model LIKE {args.model!r}: total={total}, fallback-3.0/empty={junk}, keep={total-junk}")
    if not args.apply:
        print("DRY RUN -- nothing deleted. Re-run with --apply to delete the junk rows.")
        c.close(); return 0
    c.execute(
        "DELETE FROM llm_cache WHERE model LIKE ? AND (relevance=3.0 OR TRIM(raw)='')",
        (args.model,),
    )
    c.commit()
    after = c.execute("SELECT COUNT(*) FROM llm_cache WHERE model LIKE ?", (args.model,)).fetchone()[0]
    c.execute("VACUUM")
    c.close()
    print(f"deleted {junk} rows; model rows now {after}. Re-run the pipeline to re-score them.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
