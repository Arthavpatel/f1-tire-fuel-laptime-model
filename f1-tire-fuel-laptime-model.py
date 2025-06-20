import fastf1 as f1
import pandas as pd 
import numpy as np

cache = '/Users/arthavpatel/Desktop/race_outcome_prediction/f1_cache'
f1.Cache.enable_cache(cache)

# ---------------------------------- Phase 1 ------------------------------------ #
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

# ---------------------------------- Phase 2 ------------------------------------ #
def get_driver_laps(lapdf, driver):
    if lapdf is None or lapdf.empty:
        print("Lap data is unavailable.")
        return pd.DataFrame()
    
    driver_laps = lapdf[lapdf['Driver'] == driver]
    if driver_laps.empty:
        print(f"No laps found for driver {driver}")
    return driver_laps

VER_Laps_FP1_2025 = get_driver_laps(Laps_FP1_2025, 'VER')
VER_Laps_FP2_2025 = get_driver_laps(Laps_FP2_2025, 'VER')
VER_Laps_FP3_2025 = get_driver_laps(Laps_FP3_2025, 'VER')
VER_Laps_Q_2025 = get_driver_laps(Laps_Q_2025, 'VER')
VER_Laps_R_2025 = get_driver_laps(Laps_RACE_2025,'VER')
VER_Laps_R_2024 = get_driver_laps(Laps_RACE_2024, 'VER')

# Finds wether FP2 or R has Longer stint 
def extract_stints(laps_df):
    if laps_df.empty:
        return []

    laps_df = laps_df.sort_values('LapNumber')
    stint_groups = []
    current_stint = [laps_df.iloc[0]]

    for i in range(1, len(laps_df)):
        prev_lap = laps_df.iloc[i - 1]['LapNumber']
        curr_lap = laps_df.iloc[i]['LapNumber']

        if curr_lap - prev_lap == 1:
            current_stint.append(laps_df.iloc[i])
        else:
            stint_groups.append(pd.DataFrame(current_stint))
            current_stint = [laps_df.iloc[i]]
    stint_groups.append(pd.DataFrame(current_stint)) 
    return stint_groups


def get_longest_stint(stints):
    return max(stints, key=lambda x: len(x))

soft_fp2_2025 = VER_Laps_FP2_2025[(VER_Laps_FP2_2025['Compound'] == 'SOFT') & (VER_Laps_FP2_2025['IsAccurate'] == True)]
soft_race_2025 = VER_Laps_R_2025[(VER_Laps_R_2025['Compound'] == 'SOFT') & (VER_Laps_R_2025['IsAccurate'] == True)]

stints_fp2_2025 = extract_stints(soft_fp2_2025)
stints_race_2025 = extract_stints(soft_race_2025)

longest_fp2_2025 = get_longest_stint(stints_fp2_2025) if stints_fp2_2025 else pd.DataFrame()
longest_race_2025 = get_longest_stint(stints_race_2025) if stints_race_2025 else pd.DataFrame()

if len(longest_fp2_2025) > len(longest_race_2025):
    VER_LongestStint_2025 = longest_fp2_2025
    print("🟦 FP2 has the longer Soft stint")
elif len(longest_race_2025) > len(longest_fp2_2025):
    VER_LongestStint_2025 = longest_race_2025
    print("🟥 Race has the longer Soft stint")
else:
    VER_LongestStint_2025 = longest_fp2_2025 
    print("🟨 Both stints are of equal length")

# Defining the fuel values
start_fuel = 110
FuelBurnPerLap = 1.8 

StintStartLap = VER_LongestStint_2025['LapNumber'].min()

# Calculating the fuel at the start of the stint and amount left before the stint
fuel_used_before = (StintStartLap - 1) * FuelBurnPerLap
fuel_at_start_stint = start_fuel - fuel_used_before

VER_LongestStint_2025 = VER_LongestStint_2025.copy()

# Average throttle 
avg_throttle_application = []
for i in range(len(VER_LongestStint_2025)):
    lap_number = VER_LongestStint_2025.iloc[i]['LapNumber']
    
    try:
        laps_df = VER_Laps_R_2025[VER_Laps_R_2025['LapNumber'] == lap_number].iloc[0]
        car_data = laps_df.get_car_data()

        avg_throttle = car_data['Throttle'].mean()
        
    except Exception as e:
        avg_throttle = None
        print(f"Lap {lap_number}: {e}")
    
    avg_throttle_application.append(avg_throttle)

# Average braking
avg_brake_application = []
for i in range(len(VER_LongestStint_2025)):
    lap_number = VER_LongestStint_2025.iloc[i]['LapNumber']
    try:
        laps_df = VER_Laps_R_2025[VER_Laps_R_2025['LapNumber'] == lap_number].iloc[0]
        car_data = laps_df.get_car_data()
        avg_brake = car_data['Brake'].mean()
    except Exception as e:
        avg_brake = None
        print(f"Lap {lap_number}: {e}")
    avg_brake_application.append(avg_brake)

# adding columns to the dataframe
VER_LongestStint_2025['Driver'] = 'VER'
VER_LongestStint_2025['Session'] = 'Race_2025'
VER_LongestStint_2025['FuelRemaining'] = fuel_at_start_stint - (VER_LongestStint_2025['LapNumber'] - StintStartLap) * FuelBurnPerLap
VER_LongestStint_2025['TyreAge'] = VER_LongestStint_2025['LapNumber'] - StintStartLap + 1
VER_LongestStint_2025['LapTimeSeconds'] = VER_LongestStint_2025['LapTime'].dt.total_seconds()
VER_LongestStint_2025['Sector1Seconds'] = VER_LongestStint_2025['Sector1Time'].dt.total_seconds().fillna(0)
VER_LongestStint_2025['Sector2Seconds'] = VER_LongestStint_2025['Sector2Time'].dt.total_seconds().fillna(0)
VER_LongestStint_2025['Sector3Seconds'] = VER_LongestStint_2025['Sector3Time'].dt.total_seconds().fillna(0)
VER_LongestStint_2025['AvgThrottle (%)'] = avg_throttle_application
VER_LongestStint_2025['AvgBrake (0-1)'] = avg_brake_application
baseline_time = VER_LongestStint_2025['LapTimeSeconds'].iloc[0]
VER_LongestStint_2025['LapDeltaToBase'] = VER_LongestStint_2025['LapTimeSeconds'] - baseline_time

col_to_include = [
    'Driver','Session','LapNumber', 'LapTimeSeconds', 'Compound', 'FuelRemaining', 'TyreAge', 'TrackStatus', 'Sector1Seconds',
    'Sector2Seconds', 'Sector3Seconds', 'LapDeltaToBase', 'AvgThrottle (%)', 'AvgBrake (0-1)'
]

stint_CSV_VER = VER_LongestStint_2025[col_to_include]
stint_CSV_VER.to_csv('ver_stint_data.csv', index=False)
