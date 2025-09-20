# config.py
# Project-wide constants and expected column names

REQUIRED_COLUMNS = [
    "call_id EVENT_TIME",
    "call_ts CREATE_TIME",        # timestamp string: YYYY-MM-DD HH:MM:SS
    "caller_lat latitude",
    "caller_lon longitude",
    "category EVENT_MAIN_TYPE",
    "jurisdiction station_main"
    "station_sub"
]



DATE_COL = "call_ts"
CATEGORY_COL = "category"
JURISDICTION_COL = "jurisdiction"