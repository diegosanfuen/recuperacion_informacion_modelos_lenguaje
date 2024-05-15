from tfm.descargas.DescargaBOE import DescargaBOE
import pandas as pd

BOE = DescargaBOE()
i = 0
while True:
    BOE.establecer_offset(i)
    if BOE.generar_dataset() > 100:
        break
    i += 1
BOE.obtener_dataset_final()

df = BOE.obtener_dataset_final()

pd.set_option('display.max_colwidth', None)
print(df.url)

