import sqlite3
import hashlib
import logging
import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)


# Configuración básica del logger
logging.basicConfig(filename='./mi_log.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s'
                    )


class manejador_sesiones():
    def __init__(self):
        self.id_session = 0
        self.conexion = self.obtener_db_conexion()
        self.probar_connection()
        self.conexion.close()

    def obtener_db_conexion(self):
        try:
            conn = sqlite3.connect('./sesiones.sqlite')
            logging.info("Conexion establecida")
        except Exception as e:
            logging.exception("Ocurrió un error al intentar conectar con las base de datos")

        conn.row_factory = sqlite3.Row
        return conn

    def probar_connection(self):
        try:
            conn = self.conexion
            cursor = conn.execute('SELECT * FROM sesiones limit 1').fetchall()
            logging.info("Consulta probada OK")
        except Exception as e:
            logging.exception("Ocurrió un error al intentar probar la consulta")

    def obtener_mensajes_por_sesion(self, id_session):
        prompts = []
        try:
            conn = self.obtener_db_conexion()
            res = conn.execute(f"SELECT * FROM sesiones where id_session='{id_session}';").fetchall()
            logging.info("Consulta ejecutada OK")

            for item in res:
                prompts.append(item[2])

            conn.close()
        except Exception as e:
            logging.exception(f"Ocurrió un error al obtener datos para la sesion {id_session}")

        return prompts

    def aniadir_mensajes_por_sesion(self, id_session, prompt):
        prompts = []
        try:
            conn = self.obtener_db_conexion()
            cursor = conn.cursor()
            consulta = "INSERT INTO sesiones (id_session, prompt) VALUES(?, ?);"
            cursor.execute(consulta, (id_session, prompt))
            logging.info("Consulta ejecutada OK")
            conn.commit()
            conn.close()

        except Exception as e:
            logging.exception(f"Ocurrió un error al insertar los datos para la sesion {id_session}")

        try:
            conn = self.obtener_db_conexion()
            res = conn.execute(f"SELECT * FROM sesiones where id_session='{id_session}';").fetchall()
            logging.info("Consulta ejecutada OK")

            for item in res:
                prompts.append(item[2])

            conn.close()
        except Exception as e:
            logging.exception(f"Ocurrió un error al obtener datos para la sesion {id_session}")

        return prompts