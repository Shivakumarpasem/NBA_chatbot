"""Test Google Search grounding"""
import os
from dotenv import load_dotenv
load_dotenv('.env')

api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
print(f'API key found: {bool(api_key)}')

try:
    from google import genai
    print('google-genai package: installed')
    
    client = genai.Client(api_key=api_key)
    print('Client created successfully')
    
    print('\nSending query with Google Search grounding...')
    resp = client.models.generate_content(
        model='gemini-2.0-flash',
        contents='What are the current NBA standings as of today February 27, 2026? Give me the top team in each conference with their exact win-loss record.',
        config={'tools': [{'google_search': {}}]},
    )
    print('\nResponse:')
    print(resp.text)
    
except ImportError as e:
    print(f'Import error: {e}')
    print('Run: pip install google-genai')
except Exception as e:
    print(f'Error type: {type(e).__name__}')
    print(f'Error: {e}')
