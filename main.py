import sqlite3

from knearest.knearest import KNearest


def main():
    con = sqlite3.connect("db.db")
    cur = con.cursor()

    fruits = cur.execute("SELECT * FROM fruits")

    knn = KNearest([(fruit[0], fruit[1:]) for fruit in fruits])
    print(knn.get_data_points())


if __name__ == '__main__':
    main()
