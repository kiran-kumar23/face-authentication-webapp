# check_db.py
import sqlite3

DB_PATH = "users.db"

def show_tables(c):
    print("📂 Tables in DB:")
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(c.fetchall())

def show_users(c):
    print("\n👤 Users:")
    c.execute("SELECT * FROM users;")
    rows = c.fetchall()
    if not rows:
        print("⚠️ No users found.")
    else:
        for row in rows:
            print(row)

def delete_all_users(c, conn):
    c.execute("DELETE FROM users;")
    conn.commit()
    print("✅ All users deleted (table kept).")

def drop_and_recreate_users(c, conn):
    c.execute("DROP TABLE IF EXISTS users;")
    c.execute('''
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        faiss_id INTEGER
    )
    ''')
    conn.commit()
    print("✅ Users table dropped & recreated fresh.")

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    while True:
        print("\n--- DB Manager ---")
        print("1. Show tables")
        print("2. Show users")
        print("3. Delete all users (keep table)")
        print("4. Drop & recreate users table")
        print("5. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            show_tables(c)
        elif choice == "2":
            show_users(c)
        elif choice == "3":
            delete_all_users(c, conn)
        elif choice == "4":
            drop_and_recreate_users(c, conn)
        elif choice == "5":
            break
        else:
            print("❌ Invalid choice, try again.")

    conn.close()

if __name__ == "__main__":
    main()
