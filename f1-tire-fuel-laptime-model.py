import fastf1 as f1
import pandas as pd 

cache = '/Users/arthavpatel/Desktop/race_outcome_prediction/f1_cache'
f1.Cache.enable_cache(cache)

# ---------------------------------- Loading the Grand Prix ------------------------------------ #
sessions_name = ['FP1', 'FP2', 'FP3', 'Q', 'R']
years = [2024, 2025]
grand_prix_name = 'Spanish Grand Prix'
lap_data = {}

for session_n in sessions_name:
    for year_n in years:
        try: 
            session = f1.get_session(year_n, grand_prix_name, session_n)
            session.load()
            lap_data[(session_n, year_n)] = session.laps
            print(f"Loaded {session_n} of {year_n}")
        except Exception as e:
            print(f"Failed to load {session_n} of {year_n}: {e}")


def get_laps(lap_data, session, year):
    key = (session, year)
    laps = lap_data.get(key)
    if laps is None:
        print(f"{session} data for {year} unavailable")
    return laps

Laps_RACE_2024 = get_laps(lap_data, 'R', 2024)
Laps_FP1_2025 = get_laps(lap_data, 'FP1', 2025)
Laps_FP2_2025 = get_laps(lap_data, 'FP2', 2025)
Laps_FP3_2025 = get_laps(lap_data, 'FP3', 2025)
Laps_Q_2025 = get_laps(lap_data, 'Q', 2025)
Laps_RACE_2025 = get_laps(lap_data, 'R', 2025)

