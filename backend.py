import os
# import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from langdetect import detect
from googletrans import Translator
import warnings

# warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configure Gemini API
my_key =  "AIzaSyD9fjgQqop4Nz_F_iDdIxqIykAW5Vpz5_g" 
# genai.configure(api_key=my_key)
# model = genai.GenerativeModel("gemini-1.5-flash")

def load_knowledge_base(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        return data.split('\n\n')
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

def retrieve_relevant_chunks(query, top_k=3):
    # Load knowledge base
    file_path = 'housess_knowledge_base.txt'
    knowledge_base = load_knowledge_base(file_path)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(knowledge_base)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [knowledge_base[i] for i in top_indices]

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  

def translate_text(text, src_lang='auto', target_lang='en'):
    try:
        # Language Helper Functions
        translator = Translator()
        result = translator.translate(text, src=src_lang, dest=target_lang)
        # result = result
        return result.text
    except:
        return text  # fallback to original text

# Set up LangChain LLM and Memory
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=my_key)
memory = ConversationBufferMemory()
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory)

def rag_response(query, target_lang='en'):
    original_lang = detect_language(query)

    translated_query = query

    # Translate query to English if needed
    if original_lang != 'en':
        translated_query = translate_text(query, src_lang=original_lang, target_lang='en')
    # else:
    #     translated_query = query

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

    # Combine persona, context, and conversation history
    full_context = persona + "\n\nRetrieved Context:\n" + context + "\n\n"
    # prompt = f"Context:\n{full_context}\n\nQuestion:\n{translated_query}\n\nAnswer:"

    # Use the ConversationChain to generate a response
    try:
        # response = model.generate_content(prompt)
        response = conversation.run(input=f"Context:\n{full_context}\n\nQuestion:\n{query}\n\nAnswer:")
        # answer_in_english = response.text
        answer_in_english = response.strip()
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"

    # Translate back to target language
    if target_lang != 'en':
        return translate_text(answer_in_english, src_lang='en', target_lang=target_lang)
    
    return answer_in_english

if __name__ == "__main__":
    print("Welcome to Housess AI Assistant! (Type 'exit' to quit)\n")

    while True:
        user_query = input("Ask your Queries about UAE real estate: ")

        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Thanks for using the AI assistant.")
            memory.clear()
            print("I have cleared our conversation history.")
            break

        answer = rag_response(user_query)
        print("\nBOT Response:", answer)
        print("\n" + "-"*60 + "\n")