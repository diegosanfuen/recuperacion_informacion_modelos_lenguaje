from tfm.sesiones import sesiones as ses

sesiones = ses.manejador_sesiones()

print(sesiones.obtener_mensajes_por_sesion('1234567890acbd'))