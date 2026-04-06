from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingfaceembedding
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

embedding = download_huggingfaceembedding()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name="ramayan"

# Vérifie si l’index existe
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimension de ton embedding
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # région par défaut
    )

index = pc.Index(index_name)

docsearch = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embedding)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
chain_type_kwargs = {"prompt":PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={"max_new_tokens":512,
                            'temperature':0.8})


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever= docsearch.as_retriever(search_kwargs={"k":2}),
    return_source_documents = True,
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods = ["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query":input})
    print("Response: ", result["result"])
    return str(result["result"])


if __name__ == "__main__":
    app.run(debug=True)