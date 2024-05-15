from ..sesiones import sesiones as ses

Sesion = ses.manejador_sesiones()
mensaje = Sesion.obtener_mensajes_por_sesion('1234567890acbd')
mensaje = Sesion.aniadir_mensajes_por_sesion('1234567890acbd', "Esto es otro prompt")

print(mensaje)