import pymysql

def getConnection():
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="retinal_project",
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn
