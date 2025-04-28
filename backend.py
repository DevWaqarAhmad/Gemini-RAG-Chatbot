import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

my_key = 'AIzaSyD9fjgQqop4Nz_F_iDdIxqIykAW5Vpz5_g'

genai.configure(api_key=my_key)

model = genai.GenerativeModel("gemini-1.5-flash")
response_v1a = model.generate_content("what is 2+2 formula")
# print(response_v1a.text)

file_path = 'housses  knowledge base.txt'
def load_knowledge_base(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    # Split the data into paragraphs or sentences
    chunks = data.split('\n\n')  # Adjust splitting logic based on your file format
    return chunks

knowledge_base = load_knowledge_base(file_path)

# Create a TF-IDF vectorizer for the knowledge base
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(knowledge_base)

def retrieve_relevant_chunks(query, top_k=3):
    # Vectorize the query
    query_vector = vectorizer.transform([query])
    # Compute cosine similarity between the query and all chunks
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    # Get the indices of the top-k most similar chunks
    top_indices = similarities.argsort()[-top_k:][::-1]
    # Retrieve the corresponding chunks
    relevant_chunks = [knowledge_base[i] for i in top_indices]
    return relevant_chunks

# print(retrieve_relevant_chunks("What is 2+2 formula", 3))

def rag_response(query):
    # Step 1: Retrieve relevant chunks from the knowledge base
    relevant_chunks = retrieve_relevant_chunks(query, top_k=3)

    # Handle case where no relevant chunks are found
    if not relevant_chunks:
        return ("I'm sorry, I couldn't find any relevant information to answer your question. Further more you can"
                "contact on the Number :  +971 04 569 3020 or info@housess.ae")

    context = "\n".join(relevant_chunks)

    # intro = """
    #     "input: Who are you?",
    #     "output: I am an AI agent of Housess Real Estate. I will help you choose the best Real Estate property for you.",
    # """

    persona = """
        You are a helpful AI assistant of Housess Real Estate specialized in real estate. 
        Your goal is to provide accurate, concise, and friendly responses to user queries. 
        If you don't know the answer, politely inform the user.
        """

    context = persona + context + "\n\n"
    # Step 2: Generate a response using the retrieved context
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    response = model.generate_content(prompt)

    return response.text

query = 'what are you services?'
response = rag_response(query)
print(response)

print('-----Test completed-------')