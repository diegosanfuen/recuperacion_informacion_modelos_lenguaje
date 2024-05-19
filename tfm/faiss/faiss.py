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

class manejador_faiss():

    def __init__(self):
        # Abrir y leer el archivo YAML
        with open('config/config.yml', 'r') as file:
            config = yaml.safe_load(file)

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
        index.add(vectors)
        # Guardamos los metadatos
        self.metadata = dataset[cols_metadata].to_dict('records')

    def persistir_bbdd_vectorial(self):
        # Guardar el índice de FAISS y los metadatos si es necesario
        faiss.write_index(self.index, self.ruta_base_datos)
        with open(
                self.ruta_json_metadata,
                'w') as f:
            json.dump(self.metadata, f)







