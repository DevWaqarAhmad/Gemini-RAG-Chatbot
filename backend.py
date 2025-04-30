import os
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from langdetect import detect
from googletrans import Translator

# Load environment variables
load_dotenv()

# Configure Gemini API
#my_key =  "AIzaSyD9fjgQqop4Nz_F_iDdIxqIykAW5Vpz5_g" 
genai.configure(api_key=my_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Load knowledge base
file_path = 'housess_knowledge_base.txt'

def load_knowledge_base(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        return data.split('\n\n')
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

knowledge_base = load_knowledge_base(file_path)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(knowledge_base)

def retrieve_relevant_chunks(query, top_k=3):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [knowledge_base[i] for i in top_indices]

# Language Helper Functions
translator = Translator()

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  

def translate_text(text, src_lang='auto', target_lang='en'):
    try:
        result = translator.translate(text, src=src_lang, dest=target_lang)
        return result.text
    except:
        return text  # fallback to original text

def rag_response(query, target_lang='en'):
    original_lang = detect_language(query)

    # Translate query to English if needed
    if original_lang != 'en':
        translated_query = translate_text(query, src_lang=original_lang, target_lang='en')
    else:
        translated_query = query

    relevant_chunks = retrieve_relevant_chunks(translated_query, top_k=3)

    if not relevant_chunks:
        fallback = ("I'm sorry, I couldn't find any relevant information to answer your question. "
                    "Further, you can contact us at +971 04 569 3020 or info@housess.ae.")
        if target_lang != 'en':
            return translate_text(fallback, src_lang='en', target_lang=target_lang)
        return fallback

    context = "\n".join(relevant_chunks)
    persona = (
        "You are a helpful AI assistant for Housess Real Estate specializing in real estate. "
        "Your goal is to provide accurate, concise, and friendly responses to user queries. "
        "If you don't know the answer, politely inform the user."
    )

    full_context = persona + "\n\n" + context
    prompt = f"Context:\n{full_context}\n\nQuestion:\n{translated_query}\n\nAnswer:"

    try:
        response = model.generate_content(prompt)
        answer_in_english = response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

    # Translate back to target language
    if target_lang != 'en':
        return translate_text(answer_in_english, src_lang='en', target_lang=target_lang)
    
    return answer_in_english

if __name__ == "__main__":
    print("Welcome to Housess AI Assistant! (Type 'exit' to quit)\n")

    while True:
        user_query = input("Ask your about UAE real estate: ")

        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Thanks for using the AI assistant.")
            break

        answer = rag_response(user_query)
        print("\nBOT Response:", answer)
        print("\n" + "-"*60 + "\n")
