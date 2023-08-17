

# Import necessary packages

import os
import pandas as pd
from flask import Flask, jsonify, render_template, request
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from llama_index import (LangchainEmbedding, LLMPredictor,
                         PromptHelper, ServiceContext, 
                         StorageContext,
                         load_index_from_storage)
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from transformers import pipeline

app = Flask(__name__)

class customLLM(LLM):
    model_name = "google/flan-t5-large"
    pipeline = pipeline("text2text-generation", model=model_name)

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=19999)[0]["generated_text"]
    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}
    @property
    def _llm_type(self):
        return "custom"

llm_predictor = LLMPredictor(llm=customLLM())
hfemb = HuggingFaceEmbeddings()
embed_model = LangchainEmbedding(hfemb)
max_input_size = 4096
num_output = 1000
max_chunk_overlap = 20
chunk_size_limit = 600
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

# loading the context that we had stored earlier

load_storage_context = StorageContext.from_defaults(docstore=SimpleDocumentStore.from_persist_dir(persist_dir="/app/storage"),
    vector_store=SimpleVectorStore.from_persist_dir(persist_dir="/app/storage"),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="/app/storage"),
)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model,prompt_helper=prompt_helper)
index = load_index_from_storage(storage_context=load_storage_context, service_context=service_context)
query_engine = index.as_query_engine()



@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route('/get_response', methods=['POST','GET'])
def querying():
    data = request.get_json()
    user_input = data.get("user_input", "")   
    response = query_engine.query(user_input)
    return jsonify({"response": response})



if __name__ == "__main__":
    app.run(debug=True)