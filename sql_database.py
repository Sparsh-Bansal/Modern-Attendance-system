import pymysql

db = pymysql.connect("localhost","root","12343249","sparsh" )

cursor = db.cursor()
sql = """INSERT INTO vidhi(ID,Name)
         VALUES (1,'I love you')"""
try:
   cursor.execute(sql)
   db.commit()
except:
   db.rollback()

db.close()