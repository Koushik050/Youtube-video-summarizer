import re
import requests
import json
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
from nltk.tokenize import sent_tokenize
from huggingface_hub import InferenceClient

# Hugging Face API
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_HEADERS = {"Authorization": "Bearer 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"}


# Function to extract video ID
def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


# Function to fetch the complete transcript
def fetch_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL.")

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry["text"] for entry in transcript])


# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9.,!?\s]", "", text)  # Remove special characters
    return text.strip()


# Function to chunk text into batches for LLM
def chunk_text(text, max_words=400):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


# Function for LLM summarization
def llm_summarization(text_chunks):
    summarized_chunks = []
    for chunk in text_chunks:
        payload = {"inputs": chunk, "parameters": {"max_length": 200, "min_length": 50}}
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)

        if response.status_code == 200:
            summary = response.json()[0]["summary_text"]
            summarized_chunks.append(summary)
        else:
            raise RuntimeError(f"LLM Summarization Error: {response.status_code}")
    return " ".join(summarized_chunks)


# Function to format summary as bullet points
def format_as_bullets(summary, num_bullets=5):
    sentences = sent_tokenize(summary)[:num_bullets]
    return "\n".join([f"- {sentence}" for sentence in sentences])


# Function to translate summary
def translate_summary(summary, target_lang="en"):
    return GoogleTranslator(source="en", target=target_lang).translate(summary)


# Main function
def main():
    video_url = input("Enter YouTube Video URL: ")
    num_bullets = int(input("Enter number of bullet points for summary: "))
    target_language = input("Enter output language (e.g., en, hi, te, ta): ")

    print("Fetching transcript...")
    transcript = fetch_transcript(video_url)

    print("Preprocessing transcript...")
    preprocessed_text = preprocess_text(transcript)

    print("Chunking transcript for LLM processing...")
    text_chunks = chunk_text(preprocessed_text)

    print("Generating LLM summary...")
    summarized_text = llm_summarization(text_chunks)

    print("Formatting as bullet points...")
    bullet_summary = format_as_bullets(summarized_text, num_bullets)

    print("Translating final summary...")
    translated_summary = translate_summary(bullet_summary, target_language)

    print("\n✅ Final Summary:")
    print(translated_summary)


if __name__ == "__main__":
    main()