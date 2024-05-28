# OEPIA

import sys
from pathlib import Path
import os, yaml, time
import datetime
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import gradio as gr
import logging
import secrets
import string
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import re


# Añade la ruta deseada al inicio de sys.path para priorizarla
# ruta_deseada = "/content/recuperacion_informacion_modelos_lenguaje"
# sys.path.insert(0, ruta_deseada)

import os, sys
os.environ['PROJECT_ROOT'] = r'/content/recuperacion_informacion_modelos_lenguaje/tfm'
sys.path.insert(0, os.environ['PROJECT_ROOT'])
from SesionesCollab.sesiones import ManejadorSesiones as ses
from FaissOPEIA import carga as fcg
from OEPIA.Utiles import Prompts as prompts
from OEPIA.Utiles import Utiles as utls

from SesionesCollab import sesiones as ses
from FaissOPEIACollab import carga as fcg

# Abrir y leer el archivo YAML
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config_collab.yml', 'r') as file:
    config = yaml.safe_load(file)

PATH_BASE = Path(config['ruta_base'])
directorio_proyecto = os.path.dirname(Path(PATH_BASE) / config['llm_oepia']['ruta'])
date_today = datetime.datetime.today().strftime("%Y_%m_%d")

# Configuración básica del logger
log_level = None
match config['logs_config']['level']:
    case 'DEBUG':
        log_level = logging.DEBUG
    case 'WARN':
        log_level = logging.WARNING
    case 'WARNING':
        log_level = logging.WARNING
    case 'ERROR':
        log_level = logging.ERROR
    case _:
        log_level = logging.INFO

logging.basicConfig(filename=PATH_BASE / config['logs_config']['ruta_salida_logs'] / f'logs_{date_today}.log',
                    level=log_level,
                    format=config['logs_config']['format'])

# Creamos el logger
logger = logging.getLogger()

CSS = utls.obtenerCSSOEPIAInterfaz()
try:
    modelo = config['llm_oepia']['parameters_modelo']['llm_model']
    temperature = config['llm_oepia']['parameters_modelo']['temperature']
    assistant_name = config['llm_oepia']['parameters_modelo']['nombre_asistente']
    llm = Ollama(model=modelo,
                 temperature=temperature)
except Exception as e:
    logger.error(f'Un Error se produjo al intentar cargar el modelo {modelo} : {e}')
    exit()
try:
    sesiones = ses()
except Exception as e:
    logger.error(f'Un Error se produjo al intentar cargar la base de datos de sesiones: {e}')
    exit()


token = ses.generate_token()
prompt_template = ChatPromptTemplate.from_template(prompts.obtenerPROMPTTemplatePrincipalOEPIA())
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever_inst = fcg()
retriever_faiss = retriever_inst.inialize_retriever()
retrieval_chain = create_retrieval_chain(retriever_faiss, document_chain)


def chat(pregunta):
    global token
    if("<resetea_sesion>" in pregunta.lower()):
        token = generate_token()
        answer = "Sesión reseteada"

    elif("<ver_historial>" in pregunta.lower()):
        answer = sesiones.obtener_mensajes_por_sesion(token)

    elif ("usa el agente para" in pregunta.lower()):
        try:
            response = retrieval_chain.invoke({"input": pregunta,
                                               "context": str(sesiones.obtener_mensajes_por_sesion(token))})
            answer = str(response['answer'])
            sesiones.add_mensajes_por_sesion(token, str(pregunta))
            sesiones.add_mensajes_por_sesion(token, answer)
            logger.info(str(str))
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar invocar el LLM: {e}')

    else:
        try:
            response = retrieval_chain.invoke({"input": pregunta,
                                               "context": str(sesiones.obtener_mensajes_por_sesion(token))})
            answer = str(response['answer'])
            sesiones.add_mensajes_por_sesion(token, str(pregunta))
            sesiones.add_mensajes_por_sesion(token, answer)
            logger.info(str(str))
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar invocar el LLM: {e}')

    return answer


history = ""

def format_links(text):
    # Esta función busca URLs en el texto y las reemplaza por etiquetas HTML <a>
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    formatted_text = re.sub(url_pattern, lambda
        url: f'<a href="{url.group(0)}" target="_blank" style="color: blue;">{url.group(0)}</a>', text)
    return formatted_text

with gr.Blocks() as iface:
    with gr.Row():
        texto_entrada = gr.Textbox(label="Ingresa tu mensaje", placeholder="Escribe aquí...", lines=10)
        historial_previo = gr.Textbox(label="Historial", value="", visible=False)  # Campo oculto para mantener el historial

    texto_entrada.change(fn=format_links, inputs=texto_entrada, outputs=historial_previo)

# Define los componentes de la interfaz de Gradio
# texto_entrada = gr.Textbox(label="Ingresa tu mensaje", placeholder="Escribe aquí...", lines=10)
# historial_previo = gr.Textbox(label="Historial", value="", visible=False)  # Campo oculto para mantener el historial

# Suponemos que esta función es la que maneja la comunicación con el modelo LLM
def interactuar_con_llm(texto, historial_previo):
    global history
    historial_previo = historial_previo + str(history)
    # Limpia el texto de entrada
    texto_limpio = texto.strip()

    # Simula la respuesta del modelo LLM
    respuesta = chat(texto_limpio)
    html_wrapper = f"""
    <div class="container">
        <details>
            <summary>Historial {datetime.datetime.today().strftime('%H:%M:%S')}</summary>
            <div class="content">
                <p>{history}</p>
            </div>
        </details>
    </div>
    """

    # Si es la primera interacción, no añade una línea en blanco al inicio
    if historial_previo:
        nuevo_historial = f"\n<h3><u>USUARIO:</h3></u><pre> {texto_limpio}</pre>\n\n<h3><u>OEPIA:</u></h3> <div><p>{respuesta}</p></div><br><br>{html_wrapper}\n\n"
    else:
        nuevo_historial = f"\n<h3><u>USUARIO:</u></h3><pre> {texto_limpio}</pre>\n\n<h3><u>OEPIA:</u></h3> <div><p>{respuesta}</p></div>\n\n"

    # Retorna el historial actualizado para mostrarlo en la salida
    history = nuevo_historial
    return nuevo_historial




# Esta función podría contener la lógica de postprocesamiento
def procesar_respuesta(respuesta):
    # Implementa aquí cualquier ajuste o transformación necesaria
    texto_entrada.value = ""
    return respuesta


def procesar_flag(texto_entrada, flag_option, flag_index):
    print(f"Dato marcado: {texto_entrada.value}")
    print(f"Opción seleccionada para marcar: {flag_option}")
    print(f"Índice del dato marcado: {flag_index}")




# Crea la interfaz de Gradio
iface = gr.Interface(
    fn=interactuar_con_llm,
    inputs=[texto_entrada, historial_previo],
    outputs=gr.Markdown(label="Historial de la conversación"),
    title="OEPIA: La IA especializada en ofertas de Empleo Público",
    description="Escribe un mensaje y presiona 'Submit' para interactuar con el modelo de lenguaje.",
    live=False,  # Desactiva la actualización en tiempo real
    css=CSS,
    article="Explicacion del proyecto",
    thumbnail=True,
    allow_flagging="manual",  # Permite marcar manualmente las entradas
    flagging_options=["Incorrecto", "Irrelevante", "Ofensivo"],  # Opciones para el usuario al marcar
    flagging_dir="flagged_data",  # Directorio donde se guardarán los datos marcados
)




# Inicia la interfaz
iface.launch(share=True)

