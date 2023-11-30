import sqlite3 as sqlite


def crear_bbdd_tabla():
    """Crea la tabla reto7

    Returns:
        _type_: None
    """
    con = sqlite.connect("reto7.db") # Abrir conexión
    cur = con.cursor() # Generamos un CURSOR. Necesario para ejecutar sentencias SQL
    cur.execute("""CREATE TABLE IF NOT EXISTS reto7(
        Codigo_NIF TEXT NOT NULL,
        Localidad TEXT NOT NULL
        )
    """)
    con.close()

    return None



def insertar_empresa(Codigo_NIF: str, Localidad:str):
    """Inserta datos en la tabla reto7

    Returns:
        _type_: None
    """
    con = sqlite.connect("reto7.db") # Abrir conexión
    cur = con.cursor() # Generamos un CURSOR. Necesario para ejecutar sentencias SQL
    cur.execute("INSERT INTO reto7 VALUES(?, ?)", (Codigo_NIF, Localidad))
    con.commit()
    con.close()

    return None


def consultar_todos():
    """Consulta los contenidos de la tabla reto7

    Returns:
        _type_: resultados
    """
    con = sqlite.connect("reto7.db") # Abrir conexión
    cur = con.cursor() # Generamos un CURSOR. Necesario para ejecutar sentencias SQL
    cur.execute("SELECT * FROM reto7")
    resultados = cur.fetchall()
    con.close()
    
    return resultados