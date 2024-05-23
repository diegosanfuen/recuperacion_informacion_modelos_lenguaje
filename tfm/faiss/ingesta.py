# Fuentes: https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create_retrieval_chain.html

import pandas as pd
from langchain_core.documents.base import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import pickle as pkl
import warnings
import yaml
from pathlib import Path
import logging, glob, os, datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil

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

PATH_BASE = Path(config['ruta_base'])
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


class ingesta():
    def __init__(self):
        logger.debug(f'Volcamos toda la informacion del fichero de configuracion: {config}')
        # Parametros externos configuracion
        self.embedding_llm = OllamaEmbeddings(model=config['vectorial_database']['parameters_tokenizador']['name_model_tokenizador'])
        self.ruta_db = Path(config['ruta_base']) / Path(config['vectorial_database']['ruta']) / Path(config['vectorial_database']['serialized_database'])
        logger.debug(f'Leemos la configuracion Ruta de la Base de datos: {self.ruta_db }')

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

    def generar_vector_store(self):
        text_splitter = RecursiveCharacterTextSplitter()
        documents_embd = text_splitter.split_documents(self.documentos)
        self.vector_index = FAISS.from_documents(documents_embd, self.embedding_llm)
        self.retriever = self.vector_index.as_retriever()

    def persistir_bbdd_vectorial(self):
        if os.path.exists(self.ruta_db):
            # Si existe, borra su contenido
            for filename in os.listdir(self.ruta_db):
                file_path = os.path.join(self.ruta_db, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Error al eliminar {file_path}. Razón: {e}')
        else:
            # Si no existe, crea el directorio
            os.makedirs(self.ruta_db)


        try:
            with open(self.ruta_db / Path(config['vectorial_database']['file_vector_index']), 'wb') as archivo:
                pkl.dump(self.vector_index, archivo)
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar guardar la base de datos de embbedings vector Index: {e}')

        try:
            with open(self.ruta_db / Path(config['vectorial_database']['file_retriever']), 'wb') as archivo:
                pkl.dump(self.retriever, archivo)
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar guardar la base de datos de embbedings tipo retriever: {e}')

    def inicialize_db_vect(self):
        # Leemos el contenido de la ruta de descarga del scrapper
        path_csv = Path(config['ruta_base']) / config['scrapping']['ruta'] / config['scrapping']['descarga_datos'] / '*.csv'
        path_csv_str = str(path_csv)
        archivos_csv = glob.glob(path_csv_str)
        dataframes = []
        for archivo in archivos_csv:
            # Lee el archivo CSV y añádelo a la lista de DataFrames
            df = pd.read_csv(archivo, sep='|')
            dataframes.append(df)

        self.dataset = pd.concat(dataframes, ignore_index=True)

        self.convertir_pandas_lista_documentos(self.dataset,
                                               config['scrapping']['dataset_index']['campo_texto'][0],
                                               config['scrapping']['dataset_index']['campos_metadata'])
        self.generar_vector_store()
        self.persistir_bbdd_vectorial()

    def getRetriver(self):
        return self.retriever

if __name__ == '__main__':
     BDVect = ingesta()
     BDVect.inicialize_db_vect()
     retriever = BDVect.getRetriver()

