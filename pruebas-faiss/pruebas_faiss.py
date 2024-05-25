# ChatGPT
# https://chatgpt.com/share/d293b20c-d68f-4e33-a145-57b0bc8caca2

import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import faiss
import json
import warnings
import yaml
from pathlib import Path
import logging
from transformers import BartForConditionalGeneration, BartTokenizer

# Ignorar warnings específicos de huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

# Abrir y leer el archivo YAML
with open('config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Configuración básica del logger
logging.basicConfig(filename= Path(config['ruta_salida_logs']) / 'mi_log.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s'
                    )

# Cargar el modelo y el tokenizador de BART
tokenizer = BartTokenizer.from_pretrained(config['parameters_resume']['name_model_llm'])
model = BartForConditionalGeneration.from_pretrained(config['parameters_resume']['name_model_tokenizador'])

def generate_summary(text):
    # Codificar el texto para el modelo y asegurarse de que la entrada es un tensor
    inputs = tokenizer.encode("summarize: " + text,
                              return_tensors=config['parameters_resume']['return_tensors'],
                              max_length=config['parameters_resume']['max_length'],
                              truncation=config['parameters_resume']['truncation'])

    # Generar el resumen con parámetros adecuados
    summary_ids = model.generate(inputs,
                                 max_length=config['parameters_resume']['resume']['max_length'],
                                 min_length=config['parameters_resume']['resume']['min_length'],
                                 length_penalty=config['parameters_resume']['resume']['length_penalty'],
                                 num_beams=config['parameters_resume']['resume']['num_beams'],
                                 early_stopping=config['parameters_resume']['resume']['early_stopping'])

    # Decodificar el texto generado y devolverlo
    summary = tokenizer.decode(summary_ids[0],
                               skip_special_tokens=config['parameters_resume']['resume']['skip_special_tokens'])
    return str(summary)


# Ejemplo de uso
try:
    text = "Aquí va el contenido largo del texto que quieres resumir..."
    summary = generate_summary(text)
    logging.info(summary)
except Exception as e:
    logging.exception(f"Error al generar el resumen: {e}")

# Suponemos que tu dataframe se llama df y tiene las columnas 'url', 'title', 'text'
df = pd.DataFrame({
    'url': ['https://www.boe.es/boe/dias/2024/05/09/pdfs/BOE-A-2024-9350.pdf|Orden PJC/425/2024',
            'https://www.boe.es/boe/dias/2024/05/09/pdfs/BOE-A-2024-9351.pdf|Orden PJC/426/2024'],
    'title': ['Título 1',
              'Título 2'],
    'text': ["""
    Por acuerdo adoptado por la Comisión Permanente del Consejo General del Poder Judicial, en su reunión del día 11 de abril de 2024, se ha nombrado a doña M.ª Jesús Millán de las Heras, Directora en funciones de la Escuela Judicial, con efectos desde el día 12 de abril de 2024, en consecuencia, habiéndose producido un cambio en la composición de la Comisión de Selección de las pruebas de acceso a las Carreras Judicial y Fiscal, según lo dispuesto en el artículo 305.1 de la Ley Orgánica 6/1985, de 1 de julio, del Poder Judicial, procede la publicación en el BOE de dicho nombramiento.</p>
<p class=""parrafo_2"">Madrid, 6 de mayo de 2024.–El Ministro de la Presidencia, Justicia y Relaciones con las Cortes, P. D. (Orden JUS/987/2020, de 20 de octubre), la Directora General para el Servicio Público de Justicia, Maria dels Àngels García Vidal
    ""","""
    Finalizado el plazo de presentación de instancias, y de conformidad con lo establecido en el artículo 20 Reglamento de Ingreso, Provisión de Puestos de Trabajo y Promoción Profesional del Personal Funcionario al Servicio de la Administración de Justicia, aprobado por Real Decreto 1451/2005, de 7 de diciembre,</p>
<p class=""parrafo"">Este Ministerio ha resuelto:</p>
<p class=""articulo"">Primero.</p>
<p class=""parrafo"">Aprobar las relaciones provisionales de personas aspirantes admitidas y excluidas a las pruebas selectivas para el acceso por promoción interna y sistema de concurso-oposición, al Cuerpo de Gestión Procesal y Administrativa de la Administración de Justicia, convocado por Orden PJC/104/2024, de 31 de enero, y publicar en el anexo I las relaciones provisionales de personas aspirantes excluidas, con indicación de las causas de exclusión que se relacionan en el anexo II.</p>
<p class=""articulo"">Segundo.</p>
<p class=""parrafo"">Las listas certificadas completas quedarán expuestas al público en la página web del Ministerio de la Presidencia, Justicia y Relaciones con las Cortes (www.mjusticia.es), en las páginas web de las comunidades autónomas que convoquen plazas, y en el punto de acceso general (www.administracion.gob.es).</p>
<p class=""parrafo"">En todo caso, las personas aspirantes deberán comprobar no solo que no figuran en la relación de excluidas sino, también, que sus nombres y demás datos constan correctamente en la relación de admitidas.</p>
<p class=""articulo"">Tercero.</p>
<p class=""parrafo"">Tanto las personas aspirantes excluidas como las omitidas por no figurar en las listas de admitidas ni en la de excluidas, podrán subsanar los defectos que hayan motivado su exclusión o su omisión en las listas, en el plazo de diez días hábiles, contados a partir del siguiente al de la publicación de esta orden en el «Boletín Oficial del Estado».</p>
<p class=""parrafo"">La subsanación de la solicitud o, en su caso, la modificación de los datos de la misma, deberá realizarse <em>on line</em> a través de la aplicación de Inscripción en Pruebas Selectivas (IPS) del Punto de Acceso General (https://ips.redsara.es/IPSC/secure/buscarConvocatoriasSubsanar#convocatoriasSubRef).</p>
<p class=""parrafo"">Las personas aspirantes que, dentro del plazo señalado, no subsanen la exclusión o aleguen la omisión justificando su derecho a ser incluidas en la relación de personas admitidas, serán definitivamente excluidas de la realización de las pruebas.</p>
<p class=""articulo"">Cuarto.</p>
<p class=""parrafo"">La inclusión de aspirantes en esta relación de personas admitidas a las pruebas selectivas, por el turno general y por el cupo de reserva para personas con discapacidad no supone en ningún caso el reconocimiento por parte de la Administración de que los mismos reúnen los requisitos generales o particulares exigidos en la respectiva orden de convocatoria. La acreditación y verificación de éstos tendrá lugar para los aspirantes que superen el proceso selectivo, antes de su nombramiento como funcionarios, tal y como se indica en el artículo 23 del Reglamento de Ingreso, Provisión de Puestos de Trabajo y Promoción Profesional del Personal Funcionario al Servicio de la Administración de Justicia; y en la base octava de la orden de convocatoria.
    """
             ]
})

df['text_resume'] = df['text'].apply(generate_summary)

# Cargar un modelo de BERT y su tokenizador
tokenizer = AutoTokenizer.from_pretrained(config['parameters_tokenizador']['name_model_llm'],
                                          force_download=True)
model = AutoModel.from_pretrained(config['parameters_tokenizador']['name_model_tokenizador'],
                                  force_download=True)

# Función para convertir texto en un vector usando BERT
def text_to_vector(text):
    inputs = tokenizer(text,
                       return_tensors=config['parameters_tokenizador']['return_tensors'],
                       max_length=config['parameters_tokenizador']['max_length'],
                       truncation=config['parameters_tokenizador']['truncation'],
                       padding=config['parameters_tokenizador']['padding'])
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

# Vectorizar el texto de cada fila en el dataframe
vectors = np.vstack(df['text'].apply(text_to_vector))

# Crear un índice FAISS para almacenar los vectores
dimension = vectors.shape[1]  # Dimensión de los vectores
index = faiss.IndexFlatL2(dimension)

# Añadir los vectores al índice de FAISS
index.add(vectors)

# Opcional: Guardar metadatos (url y title) para recuperarlos más tarde
metadata = df[['url', 'title', 'text_resume']].to_dict('records')

# Guardar el índice de FAISS y los metadatos si es necesario
faiss.write_index(index,
                  str(Path(config['vectorial_database']['ruta']) / config['vectorial_database']['file_indices'])
                  )


with open(str(Path(config['vectorial_database']['ruta']) / config['vectorial_database']['file_json']), 'w') as f:
    json.dump(metadata, f)


# Cargar el índice de FAISS
index = faiss.read_index(r'C:\PROYECTOS\PyCharm\pythonrun\recuperacion_informacion_modelos_lenguaje\pruebas-FaissOPEIA\FaissOPEIA\indices.index')

# Cargar metadatos (opcional)
with open(r'C:\PROYECTOS\PyCharm\pythonrun\recuperacion_informacion_modelos_lenguaje\pruebas-faiss\faiss\metadata.json', 'r') as f:
    metadata = json.load(f)



def text_to_vector(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()


# Realizar una consulta
def buscar(query, k=1):
    # Convertir la consulta en un vector
    query_vector = text_to_vector(query)

    # Asegurarse de que query_vector es un arreglo 2D con forma (1, d)
    if query_vector.ndim == 1:
        query_vector = np.expand_dims(query_vector, axis=0)

    # Buscar los k vectores más cercanos
    _, indices = index.search(query_vector, k)

    # Recuperar y mostrar los resultados
    resultados = []
    for idx in indices[0]:
        resultados.append(metadata[idx])
    return resultados


# Ejemplo de uso
resultados = buscar("Administración General")
for res in resultados:
    print(f"URL: {res['url']}, Title: {res['title']}, Text: {res['text_resume']}")
    print(res)






