# Rutina principal
from DescargaBOE import DescargaBOE
from DescargaBOCyL import DescargaBOCyL
import os

# Obtener la ruta del script actual
ruta_script = os.path.abspath("__file__")
directorio_proyecto = os.path.dirname(ruta_script)


BOE = DescargaBOE()
i = 0
while True:
    BOE.establecer_offset(i)
    if(BOE.generar_dataset() > 1000):
        break
    i += 1
BOE.obtener_dataset_final()

df = BOE.obtener_dataset_final()
df.to_csv(f'{directorio_proyecto}/datos/csv_boes_oferta_publica.csv', sep='|')

BOCyL = DescargaBOCyL()
i = 0
while True:
    BOCyL.establecer_offset(i)
    if(BOCyL.generar_dataset() > 1000):
        break
    i += 1
BOCyL.obtener_dataset_final()

df_BOCyL = BOCyL.obtener_dataset_final()
df_BOCyL.to_csv(f'{directorio_proyecto}/datos/csv_bocyls_oferta_publica.csv', sep='|')