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
            """Eres un AI llamado OEPIA, respondes a preguntas con respuestas simples,
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


import gradio as gr

history = ""


# Suponemos que esta función es la que maneja la comunicación con el modelo LLM
def interactuar_con_llm(texto, historial_previo):
    global history
    historial_previo = historial_previo + str(history)
    # Limpia el texto de entrada
    texto_limpio = texto.strip()

    # Simula la respuesta del modelo LLM
    respuesta = chat(texto_limpio)

    # Si es la primera interacción, no añade una línea en blanco al inicio
    if historial_previo:
        nuevo_historial = f"\nUSUARIO: {texto_limpio}\n\nOEPIA: {respuesta}" + f"\n\n{'*' * 50}\n\n" + historial_previo
    else:
        nuevo_historial = f"USUARIO: {texto_limpio}\n\nOEPIA: {respuesta}"

    # Retorna el historial actualizado para mostrarlo en la salida
    history = nuevo_historial
    return nuevo_historial


# Define los componentes de la interfaz de Gradio
texto_entrada = gr.Textbox(label="Ingresa tu mensaje", placeholder="Escribe aquí...", lines=10)
historial_previo = gr.Textbox(label="Historial", value="", visible=False)  # Campo oculto para mantener el historial

css = """
<style>
    .gr-textbox { 
        width: 100%; 
    }

    .interface-container {
        background-color: #000; 
        border-radius: 30px; 
    }
    button {
        background-color: #4CAF50; 
        color: dark-blue; 
        padding: 10px; 
        border: none; 
        border-radius: 20px; 
    }

    input[type='text'], textarea {
        border: 2px solid #ddd; 
        padding: 8px; 
        font-size: 16px; 
        font-color: dark-gray
    }
    label {
        color: #555; 
        font-weight: bold; 
        margin-bottom: 5px; 
    }

    .output-container {
        background-color: #DDD; 
        padding: 15px; 
        border-radius: 5px; 
    }

</style>
"""


# Esta función podría contener la lógica de postprocesamiento
def procesar_respuesta(respuesta):
    # Implementa aquí cualquier ajuste o transformación necesaria
    texto_entrada.value = ""
    return respuesta


def procesar_flag(texto_entrada, flag_option, flag_index):
    print(f"Dato marcado: {data}")
    print(f"Opción seleccionada para marcar: {flag_option}")
    print(f"Índice del dato marcado: {flag_index}")


# Crea la interfaz de Gradio
iface = gr.Interface(
    fn=interactuar_con_llm,
    inputs=[texto_entrada, historial_previo],
    outputs=gr.Textbox(label="Historial de la conversación", lines=10, interactive=False),
    title="OEPIA: La IA especializada en ofertas de Empleo Público",
    description="Escribe un mensaje y presiona 'Submit' para interactuar con el modelo de lenguaje.",
    live=False,  # Desactiva la actualización en tiempo real
    css=css,
    article="Explicacion del proyecto",
    thumbnail=True,
    allow_flagging="manual",  # Permite marcar manualmente las entradas
    flagging_options=["Incorrecto", "Irrelevante", "Ofensivo"],  # Opciones para el usuario al marcar
    flagging_dir="flagged_data",  # Directorio donde se guardarán los datos marcados
)

iface.submit_button = "Consultar"

# Inicia la interfaz
iface.launch(share=True)

