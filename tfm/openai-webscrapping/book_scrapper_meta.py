
import requests
import re 
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

import gc

from langchain_community.llms import LlamaCpp
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

from langchain.prompts import ChatPromptTemplate

targetUrl = 'https://www.ebay.es/'
response = requests.get(targetUrl)
html_text = response.text

# Removeemos cosas innecesarias
html_text = re.sub(r'<head.*?>.*?</head>', '', html_text, flags=re.DOTALL)
html_text = re.sub(r'<script.*?>.*?</script>', '', html_text, flags=re.DOTALL)
html_text = re.sub(r'<style.*?>.*?</style>', '', html_text, flags=re.DOTALL)

# limitamos los caracteres para no sobrepasarnos
# con los limites de tokens de openai
html_text = html_text[:80000]


class Libro(BaseModel):
    """Información acerca de un libro"""
    titulo:      str = Field(..., description="Titulo del libro")
    puntuacion:  str = Field(..., description="Puntuación del libro")
    precio:      str = Field(..., description="Precio del libro")

from typing import List
class LibroScrapper(BaseModel):
    """Información para extraer de HTML raw data"""
    libros: List[Libro] = Field(..., description="Lista de información de Libros")



functions = [convert_pydantic_to_openai_function(LibroScrapper)]

llm_base_path = "C:/Users/yeyos/jan/models"
llm_mistral = f"{llm_base_path}/mistral-ins-7b-q4/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
llm_llama2 = f"{llm_base_path}/mistral-ins-7b-q4/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
default_n_gpu_layers = 1  # Change this value based on your model and your GPU VRAM pool.
default_n_ctx = 3000
default_n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
default_temperature = 0.0
default_max_tokens = 2000
default_max_sentences = 3
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model = LlamaCpp(
    model_path= llm_mistral,            # Make sure the model path is correct for your system!
    n_gpu_layers=default_n_gpu_layers,
    n_batch=default_n_batch,
    n_ctx = default_n_ctx,
    temperature=default_temperature,
    max_tokens=default_max_sentences,
    top_p=1,
    verbose=False, # Verbose is required to pass to the callback manager
)


#prompt = ChatPromptTemplate.from_messages([
#    ("system", "Eres un experto en hacer web scraping y analizar HTML crudo"
#                +", si no se proporciona explícitamente no supongas"),
#    ("human", "{input}")
#])

prompt = {"ability": "Eres un experto en hacer web scraping y analizar HTML crudo",
          "input": "Hola como estas" }

chain = prompt | model

result = chain.invoke(prompt)


for libro in result:
    print(f"Título:     {libro['titulo']}")
    print(f"Puntuación: {libro['puntuacion']}")
    print(f"Precio:     {libro['precio']}")
    print("--------------------------------------------------")