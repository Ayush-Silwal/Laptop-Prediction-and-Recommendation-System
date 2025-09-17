import mysql.connector
from mysql.connector import Error

def create_connection():
    """Create a database connection to the MySQL database."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host="localhost",        
            user="root",    
            password="",
            database="laptop_database" 
        )
        if connection.is_connected():
            print("Connection to MySQL database was successful")
    except Error as e:
        print(f"Error: '{e}' occurred while connecting to MySQL database")
    return connection

def close_connection(connection):
    """Close the database connection."""
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")
