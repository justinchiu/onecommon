import sqlite3
from sqlite3 import Connection

import streamlit as st

class ResponseDB:
    DB_PATH = "response.db"
    NONE = 0
    CONFIRM = 1
    DISCONFIRM = 2
    R2S = ["none", "confirm", "disconfirm"]

    def __init__(self):
        # establish connection
        self.con = self.get_connection()
        cur = self.con.cursor()
        # create table if it doesnt already exist
        table_string = "CREATE TABLE IF NOT EXISTS responses (conv text, turn integer, agent integer, response integer)"
        cur.execute(table_string)
        self.con.commit()
        cur.close()

    @st.cache(hash_funcs={Connection: id})
    def get_connection(self):
        """Put the connection in cache to reuse if path does not change between Streamlit reruns.
        NB : https://stackoverflow.com/questions/48218065/programmingerror-sqlite-objects-created-in-a-thread-can-only-be-used-in-that-sa
        """
        return sqlite3.connect(self.DB_PATH, check_same_thread=False)

    #def __del__(self):
        #self.con.close()

    def add(self, dialogue_id, turn, agent, response):
        #cur.execute(f"insert into responses values ('{dialogue_id}', {turn}, {agent}, {response})")
        cur = self.con.cursor()
        cur.execute(
            "insert into responses values (?, ?, ?, ?)",
            (dialogue_id, turn, agent, response),
        )
        self.con.commit()
        cur.close()

    def get_id(self, dialogue_id):
        cur = self.con.cursor()
        cur.execute(
            "select * from responses where conv=:id",
            {"id": dialogue_id},
        )
        results = cur.fetchall()
        cur.close()
        return results

    def get_id_turn(self, dialogue_id, turn):
        cur = self.con.cursor()
        cur.execute(
            "select * from responses where conv=:id and turn=:turn",
            {
                "id": dialogue_id,
                "turn": turn,
            },
        )
        results = cur.fetchall()
        cur.close()
        return results

    def get_all(self):
        cur = self.con.cursor()
        cur.execute(
            "select * from responses",
        )
        results = cur.fetchall()
        cur.close()
        return results

if __name__ == "__main__":
    db = ResponseDB()
    print(db.get_id("C_d550ba11ac90479ba603ee5ec8279aae"))
    print(db.get_all())
