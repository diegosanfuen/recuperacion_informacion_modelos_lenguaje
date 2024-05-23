# Rutina principal
from DescargaBOE import DescargaBOE
from DescargaBOCyL import DescargaBOCyL
import os, datetime
import time, logging, yaml
from pathlib import Path

# Obtener la ruta del script actual
ruta_script = os.path.abspath("__file__")
os.environ['PROJECT_ROOT'] = r'C:\PROYECTOS\PyCharm\pythonrun\recuperacion_informacion_modelos_lenguaje\tfm'


# Abrir y leer el archivo YAML
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

PATH_BASE = Path(config['ruta_base'])
directorio_proyecto = os.path.dirname(Path(PATH_BASE) / config['scrapping']['ruta'])
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

BOE = DescargaBOE()
i = 0
while True:
    BOE.establecer_offset(i)
    if(BOE.generar_dataset() > 200):
        break
    time.sleep(1)
    i += 1
BOE.obtener_dataset_final()

df = BOE.obtener_dataset_final()
folder_paquete = config['scrapping']['ruta']
folder_data = config['scrapping']['descarga_datos']
df.to_csv(f'{directorio_proyecto}/{folder_paquete}/datos/csv_boes_oferta_publica.csv', sep='|')

BOCyL = DescargaBOCyL()
i = 0
while True:
    BOCyL.establecer_offset(i)
    if(BOCyL.generar_dataset() > 20):
        break
    time.sleep(1)
    i += 1
BOCyL.obtener_dataset_final()

df_BOCyL = BOCyL.obtener_dataset_final()
df.to_csv(f'{directorio_proyecto}/{folder_paquete}/datos/csv_bocyls_oferta_publica.csv', sep='|')