import mysql.connector

#establishing the connection
conn = mysql.connector.connect(user='root', host='127.0.0.1', database='test')

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

#Doping database MYDATABASE if already exists.
cursor.execute("DROP database IF EXISTS FlowchartDatabase")

#Preparing query to create a database
sql = "CREATE database FlowchartDatabase"

#Creating a database
cursor.execute(sql)

#Retrieving the list of databases
print("List of databases: ")
cursor.execute("SHOW DATABASES")
print(cursor.fetchall())

#Closing the connection
conn.close()

