from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
import gradio as gr 
import pickle

import os
os.environ['OPENAI_API_KEY'] = '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'

with open("demo_docs", "rb") as fp:
    docs_blob_pdf = pickle.load(fp)

OPENAI_API_KEY = "fe8bdc9cc4c84890a42b21266beff47a"
OPENAI_DEPLOYMENT_ENDPOINT = "https://grimaldichatgpt.openai.azure.com/" 
OPENAI_DEPLOYMENT_NAME = "gpt35t"
OPENAI_MODEL_NAME = "gpt-35-turbo"
OPENAI_DEPLOYMENT_VERSION = "2023-09-01-preview"

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = "embeddingada002"
OPENAI_ADA_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                model_name=OPENAI_MODEL_NAME,
                openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                openai_api_key=OPENAI_API_KEY,
                openai_api_type="azure",
                temperature=0)

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY')
)
vectordb = FAISS.from_documents(
    documents=docs_blob_pdf,
    embedding=embeddings, 
)
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.695}
    )

multi_query_retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever, 
    llm = llm
)
ensemble_retriever = EnsembleRetriever(
    retrievers=[multi_query_retriever_from_llm, retriever], weights=[0.50, 0.50]
)

def llm_runner() -> RetrievalQA:


    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

    template = """
    You are an Italian helpful assistant for an employee of a call center that retrieves information from the following context, which is a guidance document for employees.
    ---------------------------------------------------------
    Context: {context}

    Question: {question}
    ----------------------------------------------------------
    The answers must be in Italian. Never speak English.
    Provide URL links if it is the context.
    Do not justify your answers. Do not give information not mentioned in the CONTEXT INFORMATION.
    If you do not know the answer, just say "fornire ulteriori informazioni per la tua domanda".
    Provide a full answer, like copy paste the answer from the context.
    """
    prompt = PromptTemplate.from_template(template)

    
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                chain_type="stuff", 
                                retriever=ensemble_retriever,
                                verbose=True,
                                chain_type_kwargs = {"prompt":prompt}
                                )

    return qa

qa = llm_runner()


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def bot(history):
    res = qa(
        {
            'query': history[-1][0],
            'chat_history': history[:-1]
        }
    )
    history[-1][1] = res['result']
    return history


with gr.Blocks(title='Grimaldi', 
               theme=gr.themes.Default(primary_hue=gr.themes.colors.blue, 
                                       secondary_hue=gr.themes.colors.sky)) as demo:
    gr.Markdown("![](file/grimaldilogo.png)")
    chatbot = gr.Chatbot([],
                         avatar_images=["utente.png","bot.png"], 
                         elem_id="chatbot",
                         label='Grimaldi')
    with gr.Row():
        with gr.Column(scale=1):
            txt = gr.Textbox(
                show_label=False,
                placeholder="La prego di porre la sua domanda e premere il tasto 'Invio'",
            )


    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

if __name__ == '__main__':
    demo.launch(allowed_paths=["."])