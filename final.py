from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager

callback_manager = CallbackManager(handlers={})  # Added 'handlers' argument

n_gpu_layers = 400
n_batch = 512

def load_llm():
    llm = LlamaCpp(
        model_path="Zephyr 7B.gguf",#Model Name= 'zephyr-7b-alpha.Q4_K_M.gguf'
        n_ctx=4000,
        temperature=0.2,
        max_tokens=4000,
        top_p=1,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        callback_manager=callback_manager,
        verbose=True
    )
    return llm

# Rest of the code...

def web_load(url):
    url = "https://ellakkiaa.github.io/Kct-web_page/"
    loader = WebBaseLoader(url)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    pages = loader.load_and_split(text_splitter=text_splitter)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(pages, embeddings)
    return db

def chain():
    llm = load_llm()
    db = web_load("https://ellakkiaa.github.io/Kct-web_page/")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": "Your prompt here"},
    )

    data = qa_chain("Your question here")

    if 'result' in data:
        return data['result']
    else:
        return "Result not found in the data."

def llm_function(message, system_prompt):
    llm = load_llm()
    combined_prompt = f"{system_prompt}\n{message}"
    response = llm(combined_prompt)
    output_texts = response
    return output_texts

if __name__ == "__main__":
    system_prompt = "You are an intelligent BOT. Answer the question in message."
    message = input()

    output_text = llm_function(message, system_prompt)
    print(output_text)
