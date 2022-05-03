import sqlite3

class ResponseDB:
    DB_PATH = "response_data/response.db"

    def __init__(self):
        # establish connection
        self.con = sqlite3.connect(DB_PATH)
        cur = self.con.cursor()
        # create table if it doesnt already exist
        table_string = "CREATE TABLE IF NOT EXISTS responses (conv text, turn integer, agent integer, response integer)"
        cur.execute(table_string)

    def __del__(self):
        self.con.close()

    def add(self, dialogue_id, turn, agent, response):
        cur.execute(f"INSERT INTO responses VALUES ('{dialogue_id}', {turn}, {agent}, {response})")
        cur.commit()


