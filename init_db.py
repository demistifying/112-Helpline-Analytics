import sqlite3

def init_db():
    conn = sqlite3.connect('event_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            SL_NO INTEGER PRIMARY KEY AUTOINCREMENT,
            CREATE_TIME TEXT,
            SIGNAL_TYPE TEXT,
            EVENT_ID TEXT,
            DIALLED_NO TEXT,
            CALLER_NAME TEXT,
            EVENT_MAIN_TYPE TEXT,
            EVENT_INFORMATION TEXT,
            incident_location TEXT,
            station_main TEXT,
            station_sub TEXT,
            CALL_SIGN TEXT,
            MDT_ASSIGNED_TIME TEXT,
            DELIVERED_TIME TEXT,
            REACH_TIME TEXT,
            MDT_RESPONSE_TIME TEXT,
            ACTION_TAKEN_AT_DCC TEXT,
            CLOSURE_COMMENTS TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Run this file once before using your main app
if __name__ == '__main__':
    init_db()
