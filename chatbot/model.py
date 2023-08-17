import pickle
from flask import Flask
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTVectorStoreIndex, PromptHelper,GPTKeywordTableIndex
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LLMPredictor, ServiceContext
import torch
from langchain.llms.base import LLM
from transformers import pipeline
import pandas as pd
from llama_index import Document



# Declaring our Custom LLM Class


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
with open("KnowledgeDocument(pan_card_services).txt",encoding="utf-8") as f:
    text_extracted = f.read()
text_extracted=text_extracted.replace('\n',', ')
text_extracted=text_extracted.replace('\xa0',', ')
text_extracted=text_extracted.replace(' and ',',')
text_extracted=text_extracted.replace('-','and')

text_list = [text_extracted]

documents = [Document(text=t) for t in text_list]

# Creating and storing Indexes in Vector Database.

max_input_size = 4096
num_output = 1000
max_chunk_overlap = 20
chunk_size_limit = 600
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model,prompt_helper=prompt_helper)
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist()

