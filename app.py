import sys
import os

# Add the 'nlp' folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), "nlp"))

from nlp_llm import fetch_transcript, preprocess_text, chunk_text, llm_summarization, format_as_bullets, translate_summary
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")  # Ensure 'templates' folder exists
CORS(app)  # Enable CORS to allow frontend communication

# Route to serve the HTML file
@app.route('/')
def home():
    return render_template('MJF3.html')  # Load MJF3.html as the homepage

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        youtube_link = data.get('youtube_link')
        num_bullets = int(data.get('num_bullets', 5))
        language = data.get('language', 'en')

        if not youtube_link:
            return jsonify({'error': 'YouTube link is required'}), 400

        transcript = fetch_transcript(youtube_link)
        preprocessed_text = preprocess_text(transcript)
        text_chunks = chunk_text(preprocessed_text)
        summarized_text = llm_summarization(text_chunks)
        bullet_summary = format_as_bullets(summarized_text, num_bullets)
        translated_summary = translate_summary(bullet_summary, language)

        return jsonify({'summary': translated_summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
