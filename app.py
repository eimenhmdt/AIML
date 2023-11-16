import os
import re
import threading
import markdown
import openai
from flask import Flask, render_template, request, jsonify
from cachetools import TTLCache


app = Flask(__name__)

# OpenAI client setup
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in environment variables")
client = openai.OpenAI(api_key=api_key)


# Cache setup with TTL (e.g., 1 hour) and max size
cache = TTLCache(maxsize=1000, ttl=3600)
cache_lock = threading.Lock()


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json['text']
    output = process_text(data)
    return jsonify(output)

def process_text(text):
    lines = text.split('\n')
    processed_lines = [process_line(line) for line in lines if line.strip()]
    return '<br>'.join(processed_lines)

def cache_access(key, value=None):
    with cache_lock:
        if value:
            cache[key] = value
        return cache.get(key, None)

def openai_request(prompt, cache_key):
    try:
        response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="gpt-3.5-turbo")
        result = response.choices[0].message.content
        cache_access(cache_key, result)
        return result
    except openai.error.OpenAIError as e:
        return f"An error occurred: {str(e)}"

def process_line(line):
    if line.strip().startswith('//'):
        return ''
    elif '##' in line:
        return summarize(line.replace('##', '').strip())
    elif line.startswith('$'):
        return generic_prompt(line[1:].strip())
    elif '{{' in line and '}}' in line:
        return translate(line)
    elif line.strip().endswith('>>'):
        return continue_writing(line.rstrip('>').strip())
    else:
        return markdown.markdown(line)

def summarize(text):
    summary = cache_access(text)
    if summary:
        return summary
    return openai_request("Summarise this paragraph into one sentence: " + text, text)

def generic_prompt(prompt):
    response = cache_access(prompt)
    if response:
        return response
    return openai_request(prompt, prompt)

def translate(line):
    lang_marker = re.search(r'{{(\w{2,})}}', line)
    if lang_marker:
        target_language = lang_marker.group(1)
        text_to_translate = line.replace(lang_marker.group(0), '').strip()
        cache_key = f"{target_language}:{text_to_translate}"
        translation = cache_access(cache_key)
        if translation:
            return translation
        return openai_request(f"Translate this into {target_language}: {text_to_translate}", cache_key)

def continue_writing(text):
    completion = cache_access(text)
    if completion:
        return completion
    return openai_request(f"Continue the input text for two sentences. You must always return both, the input text and the output. This is the input: {text}", text)

if __name__ == '__main__':
    app.run(debug=True)