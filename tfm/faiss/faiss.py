import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import faiss
import json

class manejador_faiss():

    def __init__(self):
        # Parametros externos configuracion
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.ruta_base_datos = r'C:\PROYECTOS\PyCharm\pythonrun\recuperacion_informacion_modelos_lenguaje\pruebas-faiss\faiss\indices.index'
        self.ruta_json_metadata = r'C:\PROYECTOS\PyCharm\pythonrun\recuperacion_informacion_modelos_lenguaje\pruebas-faiss\faiss\metadata.json'

    def text_to_vector(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
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







