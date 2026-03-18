"""Test fallback to Google Search"""
import sys
sys.path.insert(0, '.')

from src.orchestrator import run, _gather_context

test_queries = [
    "who is the tallest NBA player ever?",
    "what team did Shaq play for?",
    "how old is LeBron James?",
    "who won rookie of the year 2024?",
]

print("Testing fallback logic:\n")
for q in test_queries:
    print(f"Query: {q}")
    ctx = _gather_context(q, [])
    has_context = bool(ctx.strip())
    print(f"  Has local context: {has_context}")
    if not has_context:
        print(f"  -> Will use Google Search")
    print()

print("="*60)
print("Testing actual response:")
print("="*60)
response = run("what team did Shaq play for?", use_llm=True, history=[])
print(response)
