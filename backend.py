
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator

from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


my_key =  "AIzaSyD9fjgQqop4Nz_F_iDdIxqIykAW5Vpz5_g" 
genai.configure(api_key=my_key)
model = genai.GenerativeModel("gemini-1.5-flash")

file_path = 'housess_knowledge_base.txt'

genai.configure(api_key=my_key)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=my_key)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

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
        return text  
chat_history = []

def rag_response(query, chat_history=[], target_lang='en'):
    original_lang = detect_language(query)

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

    # Include conversation memory (previous Q&A)
    history_text = "\n".join(chat_history)  

    full_context = f"{persona}\n\n{history_text}\n\n{context}"
    prompt = f"Context:\n{full_context}\n\nQuestion:\n{translated_query}\n\nAnswer:"

    try:
        response = model.generate_content(prompt)
        answer_in_english = response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

    chat_history.append(f"User: {translated_query}")
    chat_history.append(f"Bot: {answer_in_english}")

    if target_lang != 'en':
        return translate_text(answer_in_english, src_lang='en', target_lang=target_lang)
    
    return answer_in_english


if __name__ == "__main__":
    print("Welcome to Housess AI Assistant! (Type 'exit' to quit)\n")

    chat_history = []

    while True:
        user_query = input("Ask your about UAE real estate: ")

        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Thanks for using the AI assistant.")
            break

        answer = rag_response(user_query, chat_history)
        print("\nBOT Response:", answer)
        print("\n" + "-"*60 + "\n")
