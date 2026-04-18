import sqlite3
import os
import sys



# 推荐写法：使用绝对路径 + 自动创建文件夹
db_folder = "SalesDB"
db_file = os.path.join(db_folder, "sales.db")

# 自动创建文件夹（如果不存在）
os.makedirs(db_folder, exist_ok=True)

print(f"正在连接数据库: {db_file}")

conn = sqlite3.connect(db_file)
cursor = conn.cursor()


cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
id INTEGER PRIMARY KEY AUTOINCREMENT,
product_name TEXT NOT NULL,
quantity INTEGER NOT NULL,
price REAL NOT NULL,
sale_date TEXT NOT NULL
)""")


cursor.execute("""
INSERT INTO orders(product_name, quantity, price, sale_date) VALUES
('Product A', 10, 19.99, '2024-01-01'),
('Product B', 5, 9.99, '2024-01-02'),
('Product C', 8, 29.99, '2024-01-03')
""")

conn.commit()
conn.close()