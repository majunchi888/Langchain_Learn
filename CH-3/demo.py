import sqlite3

conn = sqlite3.connect('SalesDB/sales.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM orders")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()