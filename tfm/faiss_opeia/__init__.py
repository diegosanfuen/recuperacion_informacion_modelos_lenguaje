# Definimos una variable
import os
nombre_paquete = "faiss_opeia"


# Importamos los módulos que queremos que estén disponibles
if os.name == 'nt':
    from faiss_opeia.ingesta import ingesta
    from faiss_opeia.carga import carga
else:
    from faiss_opeia.ingesta_collab import ingesta
    from faiss_opeia.carga_collab import carga
