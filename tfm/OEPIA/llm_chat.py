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

# Introducir esta variable de entorno en el lanzador
os.environ['PROJECT_ROOT'] = r'C:\PROYECTOS\PyCharm\pythonrun\recuperacion_informacion_modelos_lenguaje\tfm'

sys.path.insert(0, os.environ['PROJECT_ROOT'])
from SesionesW import sesiones as ses
from FaissOPEIA import carga as fcg

# Abrir y leer el archivo YAML
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config.yml', 'r') as file:
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
    sesiones = ses.manejador_sesiones()
except Exception as e:
    logger.error(f'Un Error se produjo al intentar cargar la base de datos de sesiones: {e}')
    exit()

def generate_token(length=32):
    # Caracteres que pueden ser usados en el token
    characters = string.ascii_letters + string.digits
    # Generar el token
    token = ''.join(secrets.choice(characters) for _ in range(length))
    return token

token = generate_token()


prompt_template = ChatPromptTemplate.from_template("""
Te llamas OEPIA, y eres un asistente chat, tienes las siguientes misiones importantes:            
* Ayudar al usuario para encontrar las mejores ofertas de empleo público que coincidan con mi perfil. pero para ello tienes acceso a una base de datos provista por FAISS.
* Deberás de identificar las oportunidades de empleo público más relevantes que se adapten al perfil de usuario, localizando ofertas de la base de datos provista por FAISS.
* Proporciona detalles sobre los requisitos y el proceso de solicitud para cada puesto.
* Ofrece consejos sobre cómo mejorar mi aplicación y aumentar mis posibilidades de éxito.
* Cuando te pregunten por las ofertas públicas de empleo directamente otroga prioridad al los datos facilitados por la base de datos del RAG.

* Es importante que los resultados sean precisos y actualizados porque la competencia para puestos de empleo público es alta y los plazos de solicitud suelen ser estrictos. Agradezco tu ayuda en este proceso vital para mi carrera profesional.
* No te inventes información ni rellenes los datos vacios. Como eres un chat amigable :) también tienes la capacidad de reponder a preguntas no relaccionadas con las ofertas de empleo público.

<context>
{context}
</context>

Question: {input}
""")

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
            response = "Ha habido un error con el proceso ver los registros de errores"
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
    css=css,
    article="Explicacion del proyecto",
    thumbnail=True,
    allow_flagging="manual",  # Permite marcar manualmente las entradas
    flagging_options=["Incorrecto", "Irrelevante", "Ofensivo"],  # Opciones para el usuario al marcar
    flagging_dir="flagged_data",  # Directorio donde se guardarán los datos marcados
)




# Inicia la interfaz
iface.launch(share=True)

