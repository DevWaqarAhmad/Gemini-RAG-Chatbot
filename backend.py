import os
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


my_key = "AIzaSyD9fjgQqop4Nz_F_iDdIxqIykAW5Vpz5_g"

genai.configure(api_key=my_key)

model = genai.GenerativeModel("gemini-1.5-flash")

file_path = 'housess_knowledge_base.txt'

def load_knowledge_base(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        
        chunks = data.split('\n\n')
        return chunks
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
    relevant_chunks = [knowledge_base[i] for i in top_indices]
    return relevant_chunks

def rag_response(query):
    relevant_chunks = retrieve_relevant_chunks(query, top_k=3)

    if not relevant_chunks:
        return ("I'm sorry, I couldn't find any relevant information to answer your question. "
                "Further, you can contact us at +971 04 569 3020 or info@housess.ae.")

    context = "\n".join(relevant_chunks)

    persona = (
        "You are a helpful AI assistant for Housess Real Estate specializing in real estate. "
        "Your goal is to provide accurate, concise, and friendly responses to user queries. "
        "If you don't know the answer, politely inform the user."
    )

    full_context = persona + "\n\n" + context

    prompt = f"Context:\n{full_context}\n\nQuestion:\n{query}\n\nAnswer:"

    response = model.generate_content(prompt)

    return response.text

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
