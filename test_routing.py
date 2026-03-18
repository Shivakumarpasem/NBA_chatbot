"""Test query routing"""
import sys
sys.path.insert(0, '.')

from src.orchestrator import _is_live_web_query, run

test_queries = [
    "who is leading",
    "standings",
    "who is leading in NBA?",
    "NBA standings",
    "what about oklahoma",
]

print("Testing query detection:")
for q in test_queries:
    is_live = _is_live_web_query(q)
    print(f"  '{q}' -> is_live_web_query: {is_live}")

print("\n" + "="*60)
print("Testing actual response for 'who is leading':")
print("="*60)

response = run("who is leading in the NBA?", use_llm=True, history=[])
print(response)
