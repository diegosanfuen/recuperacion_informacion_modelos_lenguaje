import requests
import pdfplumber
import io


class Prompts:

    @staticmethod
    def obtenerPROMPTTemplatePrincipalOEPIA():
        return """
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
            """


class Utiles:
    @staticmethod
    def obtenerPROMPTTemplatePrincipalOEPIA():
        return """
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
            """

    @staticmethod
    def obtenerCSSOEPIAInterfaz():
        return """
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
    @staticmethod
    def obtener_boe_texto(url):
        response = requests.get(url)

        # Asegurarse de que la solicitud fue exitosa
        if response.status_code == 200:
            # Abrir el PDF desde un stream de bytes
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text = ''
                # Extraer texto de cada página
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        else:
            print("Error al descargar el archivo")
        return text
