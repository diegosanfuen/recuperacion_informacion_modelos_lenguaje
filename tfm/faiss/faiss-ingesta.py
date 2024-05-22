# Fuentes: https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create_retrieval_chain.html

import pandas as pd
from langchain_core.documents.base import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import picke as pkl
import numpy as np
from transformers import AutoModel, AutoTokenizer
import faiss
import json
import warnings
import yaml
from pathlib import Path
import logging
from transformers import BartForConditionalGeneration, BartTokenizer
import os
import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PROJECT_ROOT'] = r'C:\PROYECTOS\PyCharm\pythonrun\recuperacion_informacion_modelos_lenguaje\tfm'

# Ignorar warnings específicos de huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

# Abrir y leer el archivo YAML
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

class manejador_faiss():
    def __init__(self):
        fecha_hoy = datetime.datetime.today().strftime("%Y_%m_%d")
        logging.debug(f'Volcamos toda la informacion del fichero de configuracion: {config}')
        # Configuración básica del logger
        logging.basicConfig(filename=Path(config['ruta_salida_logs']) / f'logs_{fecha_hoy}.log',
                            level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Parametros externos configuracion
        self.embedding_llm = OllamaEmbeddings(model="llama3")
        self.tokenizer = AutoTokenizer.from_pretrained(config['parameters_tokenizador']['name_model_llm'],
                                                       force_download=True)
        self.model = AutoModel.from_pretrained(config['parameters_tokenizador']['name_model_tokenizador'],
                                               force_download=True)

        self.ruta_base_datos = Path(config['vectorial_database']['ruta']) / config['vectorial_database']['file_indices']
        self.ruta_json_metadata = Path(config['vectorial_database']['ruta']) / config['vectorial_database']['file_json']
        logging.debug(f'Leemos la configuracion Ruta de la Base de datos: {self.ruta_base_datos}')

    def convertir_pandas_lista_documentos(self, dataframe, col_text, cols_metadata):
        # Lista para almacenar los documentos
        documentos = []

        # Iterar sobre cada fila del DataFrame
        for index, row in dataframe.iterrows():
            # Crear un objeto Document
            doc = Document(
                page_content=row[col_text],  # El contenido principal del documento
                metadata={
                    campo: row[campo] for campo in cols_metadata
                }
            )
            # Añadir el Documento a la lista
            documentos.append(doc)

        self.documentos = documentos

    def generar_vecrtor_store(self):
        text_splitter = RecursiveCharacterTextSplitter()
        documents_embd = text_splitter.split_documents(self.documentos)
        self.vector_index = FAISS.from_documents(documents_embd, self.embeddings_llm)
        self.retriever = self.vector_index.as_retriever()




    def text_to_vector(self, text):
        inputs = self.tokenizer(text,
                                return_tensors=config['parameters_tokenizador']['return_tensors'],
                                max_length=config['parameters_tokenizador']['max_length'],
                                truncation=config['parameters_tokenizador']['truncation'],
                                padding=config['parameters_tokenizador']['padding'])
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

    def vectorizar(self, dataset, col_text, cols_metadata):
        try:
            # Vectorizar el texto de cada fila en el dataframe
            self.vectors = np.vstack(dataset[col_text].apply(self.text_to_vector))
            # Crear un índice FAISS para almacenar los vectores
            dimension = self.vectors.shape[1]  # Dimensión de los vectores
            print(dimension)
            self.index = faiss.IndexFlatL2(dimension)
            # Añadir los vectores al índice de FAISS
            self.index.add(self.vectors)
            # Guardamos los metadatos
            self.metadata = dataset[cols_metadata].to_dict('records')
        except Exception as e:
            logging.error(f'Un Error se produjo al intentar generar los embbedings: {e}')

    def persistir_bbdd_vectorial(self):
        try:
            with open(r'C:\PROYECTOS\PyCharm\pythonrun\recuperacion_informacion_modelos_lenguaje\tfm\faiss\db\retriever.pkl', 'wb') as archivo:
                pkl.dump(self.vector_index, archivo)
        except Exception as e:
            logging.error(f'Un Error se produjo al intentar guardar la base de datos de embbedings: {e}')

    def cargar_bbdd_vectorial(self):
        try:
            # Carga la base de datos vectorial configurada en el fichero de config.yml
            # Cargar el índice de FAISS
            self.index = None
            self.metadata = None
            self.index = faiss.read_index(str(self.ruta_base_datos))

            # Cargar metadatos (opcional)
            with open(self.ruta_json_metadata,
                    'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            logging.error(f'Base de datos: {self.ruta_json_metadata} y {self.ruta_base_datos}')
            logging.error(f'Un Error se produjo al intentar cargar la base de datos de embbedings: {e}')

    def buscar_bbdd_vectorial(self, q, k=3):
        # Consulta a la BBDD Vectorial y obientre k resultados más proximos
        # Convertir la consulta en un vector
        query_vector = self.text_to_vector(q)

        # Asegurarse de que query_vector es un arreglo 2D con forma (1, d)
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        # Buscar los k vectores más cercanos
        _, indices = self.index.search(query_vector, k)


        # Recuperar y mostrar los resultados
        self.resultados = []
        for idx in indices[0]:
            self.resultados.append(self.metadata[idx])
        return self.resultados

    def crear_bbdd_vectorial(self, dataset, col_text, cols_metadata):
        self.vectorizar(dataset, col_text, cols_metadata)
        self.persistir_bbdd_vectorial()


if __name__ == '__main__':
    BDVect = manejador_faiss()
    # df = pd.read_csv(r'C:\PROYECTOS\PyCharm\pythonrun\recuperacion_informacion_modelos_lenguaje\tfm\ObtencionDatos\datos\csv_boes_oferta_publica.csv', sep='|')
    # df['embbeding'] = df['texto'].iloc[1:400].apply(BDVect.text_to_vector)
    # BDVect.vectorizar(df, 'texto', ['url', 'titulo'])
    # BDVect.persistir_bbdd_vectorial()
    BDVect.cargar_bbdd_vectorial()
    print("Búsqueda Arquitecto")
    print(BDVect.buscar_bbdd_vectorial("Arquitecto", 10))
    print("Búsqueda Informático")
    print(BDVect.buscar_bbdd_vectorial("Informático", 10))
    print("Búsqueda Policía")
    print(BDVect.buscar_bbdd_vectorial("Policía", 10))











