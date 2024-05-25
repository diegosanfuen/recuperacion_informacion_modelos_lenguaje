# Definimos una variable
import sys, os
from pathlib import Path
nombre_paquete = "faiss_opeia"
root_path = Path(os.environ['PROJECT_ROOT'])
sys.path.insert(0, root_path / nombre_paquete)


# Importamos los módulos que queremos que estén disponibles
if os.name == 'nt':
    from faiss_opeia.ingesta import ingesta
    from faiss_opeia.carga import carga
else:
    from faiss_opeia.ingesta import ingesta_collab
    from faiss_opeia.carga import carga_collab
