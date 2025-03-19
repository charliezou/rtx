import sqlite3
import json
from datetime import datetime
import numpy as np

class VoiceCardDB:
    def __init__(self, db_path='voice_cards.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS voice_cards
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 audio_path TEXT NOT NULL,
                 mfcc_features TEXT NOT NULL,
                 base_freq REAL,
                 formants TEXT,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            
            self.conn.execute('''CREATE TABLE IF NOT EXISTS recognition_logs
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 input_features TEXT NOT NULL,
                 match_card_id INTEGER,
                 similarity_score REAL,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    def add_voice_card(self, name, audio_path, mfcc_features, base_freq, formants):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''INSERT INTO voice_cards 
                (name, audio_path, mfcc_features, base_freq, formants)
                VALUES (?,?,?,?,?)''',
                (name, audio_path, json.dumps(mfcc_features.tolist()), 
                 base_freq, json.dumps(formants)))
            return cursor.lastrowid

    def delete_voice_card(self, card_id):
        with self.conn:
            self.conn.execute("DELETE FROM voice_cards WHERE id=?", (card_id,))
            

    def get_all_voice_cards(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM voice_cards')
        rows = cursor.fetchall()
        return [{
            'id': row[0],
            'name': row[1],
            'mfcc_features': np.array(json.loads(row[3])),
            'base_freq': row[4],
            'formants': json.loads(row[5])
        } for row in rows]

    def log_recognition(self, input_features, match_card_id=None, similarity_score=None):
        with self.conn:
            self.conn.execute('''INSERT INTO recognition_logs
                (input_features, match_card_id, similarity_score)
                VALUES (?,?,?)''',
                (json.dumps(input_features.tolist()), match_card_id, similarity_score))

    def __del__(self):
        self.conn.close()