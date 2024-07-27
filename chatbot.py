import gradio as gr
import os
import codecs
import openai
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
# from dotenv import load_dotenv

# load_dotenv()

pc = Pinecone(api_key=os.getenv("Pinecone_API_KEY"))
cloud = os.environ.get("PINECONE_CLOUD") or "aws"
region = os.environ.get("PINECONE_REGION") or "us-east-1"

spec = ServerlessSpec(cloud=cloud, region=region)

# Set up OpenAI API
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY")
)
# Load the Vietnamese embedding model
model = SentenceTransformer("dangvantuan/vietnamese-embedding")

# Function to get embeddings


def get_embeddings(sentences):
    tokenizer_sent = [tokenize(sent) for sent in sentences]
    embeddings = model.encode(tokenizer_sent)
    return embeddings


# Function to create metadata


def create_meta_batch(sentences):
    return [{"text": sent} for sent in sentences]


# Function to read sentences from a file


def read_sentences_from_file(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as file:
        sentences = file.readlines()
    # Remove any leading/trailing whitespace
    sentences = [sentence.strip()
                 for sentence in sentences if sentence.strip()]
    return sentences


# rag-index, new_rag-index
def process_and_upsert(sentences, batch_size=100, index_name="new-rag-index"):
    print(pc.list_indexes().names())
    if index_name not in pc.list_indexes().names():
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=768,  # dimensionality of text-embedding-ada-002
            metric="cosine",
            spec=spec,
        )

        # connect to index
        index = pc.Index(index_name)

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            embeddings = get_embeddings(batch)
            ids = [str(i + j) for j in range(len(batch))]
            meta_batch = create_meta_batch(batch)
            vectors = [
                {"id": id_, "values": embedding.tolist(), "metadata": meta}
                for id_, embedding, meta in zip(ids, embeddings, meta_batch)
            ]
            index.upsert(vectors)
    else:
        index = pc.Index(index_name)
    return index


# Initialize vector database
file_path = "data.txt"
sentences = read_sentences_from_file(file_path)
index = process_and_upsert(sentences, batch_size=100,
                           index_name="new-rag-index")

# A dictionary to store conversation history for each session
conversation_history = {}

# Define the system prompt with rules
system_prompt = """
You are a chatbot designed to act as a real estate consulting assistant. Your main role is to provide users with accurate and useful information, advice, and insights on real estate queries. When answering questions, you must adhere to the following principles:

1. Accuracy and Relevance: Ensure your answers are based on current and relevant real estate data and trends, but do not reference or refer to specific data sources in your responses.
2. Scope Management: If a query is beyond your capabilities or tools, instruct users on how to find additional help or suggest alternative methods to find the information they require.
3. Scope Management: If a query is beyond your capabilities or tools, instruct users on how to find additional help or suggest alternative methods to find the information they require.
4. Information Gathering: Ask for additional information if it is necessary to provide a more accurate answer.
5. Language Consistency: Always reply in the user's language. If the user speaks Vietnamese, you should reply in Vietnamese.

I. Identifying User Needs:
1. Ask the user what type of real estate service they are interested in (buying, selling, renting, or investing).
2. Gather basic information about their requirements (location, budget, property type, etc.).

II. Providing Information and Options
1. Based on the user’s requirements, provide information on available properties or services.
2. Share links or details of properties that match their criteria.
3. Offer to schedule viewings or consultations if applicable.

III. Answering Questions
1. Be prepared to answer common questions related to real estate transactions (e.g., mortgage rates, property taxes, neighborhood details, etc.).
2. Provide clear and concise answers.
Example:
"We have several properties that match your criteria. Here are a few options: [Property 1 Details], [Property 2 Details], [Property 3 Details]. Would you like to schedule a viewing or need more information on any of these?"

General Tips:
1. Maintain a friendly and professional tone throughout the conversation.
2. Be concise and to the point, avoiding overly technical jargon.
3. Ensure quick and accurate responses to user queries.
4. Offer personalized assistance based on user inputs.
5. Ensure user data privacy and confidentiality at all times.

You will be provided with additional documents containing information on properties. Use these documents to answer questions based on the provided data.  Distance is calculated from the location of the house to the city center.
If the document does not contain the information needed to answer a question, simply write:
"Tôi không có thông tin về vấn đề này."
"""


def first_conversational_rag(session_id):
    conversation_history[session_id] = {
        "messages": [{"role": "system", "content": system_prompt}],
        "info": {},
    }


def conversational_rag(session_id, question, history):
    limit = 3750
    # res = openai.Embedding.create(
    #     input=[question],
    #     engine="text-embedding-ada-002"
    # )
    # xq = res['data'][0]['embedding']
    tokenizer_sent = [tokenize(question)]
    xq = model.encode(tokenizer_sent)
    xq = xq.tolist()[0]
    contexts = []
    time_waited = 0
    while len(contexts) < 3 and time_waited < 60 * 5:
        res = index.query(vector=xq, top_k=3, include_metadata=True)
        contexts = contexts + [x["metadata"]["text"] for x in res["matches"]]
        print(f"Retrieved {contexts}")
        time.sleep(2)
        time_waited += 20
    if time_waited >= 60 * 5:
        print("Timed out waiting for contexts to be retrieved.")
        contexts = [
            "No documents retrieved. Try to answer the question yourself!"]
    for i in range(1, len(contexts)):
        if len("\n".join(contexts[:i])) >= limit:
            prompt = "\n".join(contexts[: i - 1])
            break
        elif i == len(contexts) - 1:
            prompt = "\n".join(contexts)
    # Include stored information in the context
    stored_history = conversation_history[session_id]["messages"]
    stored_history.append(
        {"role": "system", "content": "Additional document:" + prompt}
    )
    stored_history.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=stored_history, max_tokens=350, temperature=0.7
    )
    answer = response.choices[0].message.content
    conversation_history[session_id]["messages"].append(
        {"role": "user", "content": question}
    )
    conversation_history[session_id]["messages"].append(
        {"role": "assistant", "content": answer}
    )
    history.append((question, answer))
    return "", history


def clear_history(session_id):
    if session_id in conversation_history:
        del conversation_history[session_id]
    first_conversational_rag(session_id)
    gr.Info("Conversation history cleared.")
    return


def open_ui(session_id):
    if session_id not in conversation_history:
        first_conversational_rag(session_id)
        return gr.update(visible=True)
    return None, None


def main():
    with gr.Blocks() as demo:
        session_id = gr.Textbox(label="Enter the name of the conversation:")
        start_button = gr.Button("Start Conversation", variant="primary")

        with gr.Column(visible=False) as main_row:
            chatbot = gr.Chatbot(
                value=[
                    [
                        None,
                        "Xin chào, Chào mừng bạn đến với công ty bất động sản ABC. Tôi là trợ lý ảo ở đây để giúp bạn giải quyết các nhu cầu về bất động sản. Hôm nay tôi có thể hỗ trợ bạn như thế nào?",
                    ]
                ],
                label="Chatbot",
                placeholder="Chatbot is ready to answer your questions.",
            )
            question = gr.Textbox(label="Your Question")
            # submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear Conversation History")

            question.submit(
                conversational_rag,
                inputs=[session_id, question, chatbot],
                outputs=[question, chatbot],
            )
            clear_btn.click(clear_history, inputs=[session_id], outputs=[])

        start_button.click(open_ui, inputs=[session_id], outputs=[main_row])

    demo.launch()


if __name__ == "__main__":
    main()
