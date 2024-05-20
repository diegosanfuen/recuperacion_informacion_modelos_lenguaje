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
import os

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
        print(config)
        # Configuración básica del logger
        logging.basicConfig(filename=Path(config['ruta_salida_logs']) / 'mi_log.log',
                            level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Parametros externos configuracion
        self.tokenizer = AutoTokenizer.from_pretrained(config['parameters_tokenizador']['name_model_llm'],
                                                       force_download=True)
        self.model = AutoModel.from_pretrained(config['parameters_tokenizador']['name_model_tokenizador'],
                                               force_download=True)

        self.ruta_base_datos = Path(config['vectorial_database']['ruta']) / config['vectorial_database']['file_indices']
        self.ruta_json_metadata = Path(config['vectorial_database']['ruta']) / config['vectorial_database']['file_json']

    def text_to_vector(self, text):
        inputs = self.tokenizer(text,
                                return_tensors=config['parameters_tokenizador']['return_tensors'],
                                max_length=config['parameters_tokenizador']['max_length'],
                                truncation=config['parameters_tokenizador']['truncation'],
                                padding=config['parameters_tokenizador']['padding'])
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

    def vectorizar(self, dataset, col_text, cols_metadata):
        # Vectorizar el texto de cada fila en el dataframe
        self.vectors = np.vstack(dataset[col_text].apply(self.text_to_vector))
        # Crear un índice FAISS para almacenar los vectores
        dimension = self.vectors.shape[1]  # Dimensión de los vectores
        self.index = faiss.IndexFlatL2(dimension)
        # Añadir los vectores al índice de FAISS
        self.index.add(self.vectors)
        # Guardamos los metadatos
        self.metadata = dataset[cols_metadata].to_dict('records')

    def persistir_bbdd_vectorial(self):
        # Guardar el índice de FAISS y los metadatos si es necesario
        faiss.write_index(self.index, self.ruta_base_datos)
        with open(
                self.ruta_json_metadata,
                'w') as f:
            json.dump(self.metadata, f)

    def cargar_bbdd_vectorial(self):
        # Carga la base de datos vectorial configurada en el fichero de config.yml
        # Cargar el índice de FAISS
        self.index = None
        self.metadata = None
        self.index = faiss.read_index(self.ruta_base_datos)

        # Cargar metadatos (opcional)
        with open(self.ruta_json_metadata,
                'r') as f:
            self.metadata = json.load(f)

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
    df = pd.read_csv(r'C:\PROYECTOS\PyCharm\pythonrun\recuperacion_informacion_modelos_lenguaje\tfm\ObtencionDatos\datos\csv_boes_oferta_publica.csv', sep='|')
    df['embbeding'] = df['texto'].apply(BDVect.text_to_vector)
    BDVect.vectorizar(df, 'texto', ['url', 'tittle'])
    BDVect.persistir_bbdd_vectorial()











