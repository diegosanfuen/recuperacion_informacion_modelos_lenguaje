# Definimos una variable
import sys, os
from pathlib import Path
nombre_paquete = "faiss_opeia"
root_path = Path(os.environ['PROJECT_ROOT'])
sys.path.insert(0, root_path / nombre_paquete)


# Importamos los módulos que queremos que estén disponibles
if os.name == 'nt':
    from .ingesta import ingesta
    from .carga import carga
else:
    from .ingesta_collab import ingesta
    from .carga_collab import carga
