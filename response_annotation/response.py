import sqlite3
from sqlite3 import Connection

import streamlit as st

class ResponseDB:
    DB_PATH = "response_data/response.db"

    def __init__(self):
        # establish connection
        self.con = self.get_connection()
        cur = self.con.cursor()
        # create table if it doesnt already exist
        table_string = "CREATE TABLE IF NOT EXISTS responses (conv text, turn integer, agent integer, response integer)"
        cur.execute(table_string)

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
        cur.commit()

    def get_id(self, dialogue_id):
        cur = self.con.cursor()
        cur.execute(
            "select * from responses where conv=:id",
            {"id": dialogue_id},
        )
        results = cur.fetchall()
        return results
