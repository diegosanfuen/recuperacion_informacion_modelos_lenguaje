# OEPIA
from tfm.sesiones import sesiones as ses

import logging
import gc

from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import gradio as gr

llm = Ollama(model='llama3')
sesiones = ses.manejador_sesiones()
mensaje = sesiones.obtener_mensajes_por_sesion('1234567890acbd')


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un AI llamado AIcia, respondes a preguntas con respuestas simples,
            además debes contestar de vuelta preguntas acorde al contexto, eres especialista
            en ofertas de empleo público, todas las respuestas las debes
            de dar asocidas a dicho tema""",
        ),

        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

chain = prompt_template | llm

def chat(pregunta):
    response = chain.invoke({"input": pregunta, "chat_history": sesiones.obtener_mensajes_por_sesion('1234567890acbd')})
    sesiones.add_mensajes_por_sesion('1234567890acbd', str(HumanMessage(content=pregunta)))
    sesiones.add_mensajes_por_sesion('1234567890acbd', str(AIMessage(content=response)))
    print(str(AIMessage(content=response)))
    return response


iface = gr.Interface(fn=chat, inputs="text", outputs="text")
iface.launch()
