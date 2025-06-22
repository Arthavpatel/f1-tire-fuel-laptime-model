import fastf1 as f1
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

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
start_fuel = 110
FuelBurnPerLap = 1.8
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

def compare_and_process_stint(driver_laps_fp, driver_laps_race, compound, filename):
    fp2_laps = driver_laps_fp[(driver_laps_fp['Compound'] == compound) & (driver_laps_fp['IsAccurate'] == True)]
    race_laps = driver_laps_race[(driver_laps_race['Compound'] == compound) & (driver_laps_race['IsAccurate'] == True)]

    stint_fp2 = extract_stints(fp2_laps)
    stint_race = extract_stints(race_laps)

    longest_fp2 = get_longest_stint(stint_fp2) if stint_fp2 else pd.DataFrame()
    longest_race = get_longest_stint(stint_race) if stint_race else pd.DataFrame()
    
    if len(longest_fp2) > len(longest_race):
        selected_stint = longest_fp2
        source = 'FP2'
    elif len(longest_race) > len(longest_fp2):
        selected_stint = longest_race
        source = 'Race'
    else:
        selected_stint = longest_fp2
        source = 'Equal / FP2 Default'   
    print(f" {compound}: Longer stint from {source} with {len(selected_stint)} laps.")

    if selected_stint.empty:
        return 
    
    stint_start_lap = selected_stint['LapNumber'].min()
    fuel_at_start_stint = start_fuel - (stint_start_lap - 1) * FuelBurnPerLap

    avg_throttle = []
    avg_brake = []

    for i in range(len(selected_stint)):
        LapNumber = selected_stint.iloc[i]['LapNumber']
        try:
            lap_data = VER_Laps_R_2025[VER_Laps_R_2025['LapNumber'] == LapNumber].iloc[0]
            car_data = lap_data.get_car_data()
            avg_throttle.append(car_data['Throttle'].mean())
            avg_brake.append(car_data['Brake'].mean())
        except Exception as e:
            avg_throttle.append(None)
            avg_brake.append(None)
            print(f"{compound} lap {LapNumber} error : {e}")
    selected_stint = selected_stint.copy()
    selected_stint['Driver'] = 'VER'
    selected_stint['Session'] = source + '_2025'
    selected_stint['FuelRemaining'] = fuel_at_start_stint - (selected_stint['LapNumber'] - stint_start_lap) * FuelBurnPerLap
    selected_stint['TyreAge'] = selected_stint['LapNumber'] - stint_start_lap + 1
    selected_stint['LapTimeSeconds'] = selected_stint['LapTime'].dt.total_seconds()
    selected_stint['Sector1Seconds'] = selected_stint['Sector1Time'].dt.total_seconds().fillna(0)
    selected_stint['Sector2Seconds'] = selected_stint['Sector2Time'].dt.total_seconds().fillna(0)
    selected_stint['Sector3Seconds'] = selected_stint['Sector3Time'].dt.total_seconds().fillna(0)
    selected_stint['AvgThrottle (%)'] = avg_throttle
    selected_stint['AvgBrake (0-1)'] = avg_brake
    baseline_time = selected_stint['LapTimeSeconds'].iloc[0]
    selected_stint['LapDeltaToBase'] = selected_stint['LapTimeSeconds'] - baseline_time

    cols = [
        'Driver','Session','LapNumber', 'LapTimeSeconds', 'Compound', 'FuelRemaining', 'TyreAge', 'TrackStatus',
        'Sector1Seconds','Sector2Seconds','Sector3Seconds','LapDeltaToBase','AvgThrottle (%)','AvgBrake (0-1)'
    ]
    selected_stint[cols].to_csv(filename, index=False)
    print(f"✅ Saved {compound} stint to {filename}")
    return selected_stint

VER_LongestStint_SOFT_2025 = compare_and_process_stint(VER_Laps_FP2_2025, VER_Laps_R_2025, 'SOFT', 'VER_SOFT_STINT.csv')
VER_LongestStint_MEDIUM_2025 = compare_and_process_stint(VER_Laps_FP2_2025, VER_Laps_R_2025, 'MEDIUM', 'VER_MEDIUM_STINT.csv')
VER_LongestStint_HARD_2025 = compare_and_process_stint(VER_Laps_FP2_2025, VER_Laps_R_2025, 'HARD', 'VER_HARD_STINT.csv')

# ---------------------------------- Phase 3 ------------------------------------ #

def train_poly_model(stint_df, degree = 2):
    features = ['TyreAge', 'FuelRemaining', 'AvgThrottle (%)', 'AvgBrake (0-1)']
    X = stint_df[features].fillna(0)
    y = stint_df['LapTimeSeconds']
    poly_model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(2)),
        ('model', LinearRegression())
    ])
    poly_model.fit(X,y)
    return poly_model

model_soft = train_poly_model(VER_LongestStint_SOFT_2025)
model_medium = train_poly_model(VER_LongestStint_MEDIUM_2025)
model_hard = train_poly_model(VER_LongestStint_HARD_2025)
features = ['TyreAge', 'FuelRemaining', 'AvgThrottle (%)', 'AvgBrake (0-1)']

# SOFT
X_SOFT = VER_LongestStint_SOFT_2025[features].fillna(0)
VER_LongestStint_SOFT_2025['PredictedLapTime'] = model_soft.predict(X_SOFT)
VER_LongestStint_SOFT_2025.to_csv("ver_stint_data_with_prediction_SOFT.csv", index=False)

# MEDIUM
X_MEDIUM = VER_LongestStint_MEDIUM_2025[features].fillna(0)
VER_LongestStint_MEDIUM_2025['PredictedLapTime'] = model_medium.predict(X_MEDIUM)
VER_LongestStint_MEDIUM_2025.to_csv("ver_stint_data_with_prediction_MEDIUM.csv", index=False)

X_HARD = VER_LongestStint_HARD_2025[features].fillna(0)
VER_LongestStint_HARD_2025['PredictedLapTime'] = model_hard.predict(X_HARD)
VER_LongestStint_HARD_2025.to_csv("ver_stint_data_with_prediction_HARD.csv", index=False)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(VER_LongestStint_SOFT_2025['LapNumber'], VER_LongestStint_SOFT_2025['LapTimeSeconds'], marker='o', label='Actual Lap Time')
plt.plot(VER_LongestStint_SOFT_2025['LapNumber'], VER_LongestStint_SOFT_2025['PredictedLapTime'], marker='x', linestyle='--', label='Predicted Lap Time')
plt.xlabel('Lap Number')
plt.ylabel('Lap Time (seconds)')
plt.title('Actual vs Predicted Lap Time for VER Soft Stint (Spanish GP 2025)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

# ---------------------------------- Phase 4 ------------------------------------ #

def simulate_stint_and_plot(model, stint_length, start_fuel, fuel_burn_per_lap, reference_df, title_suffix=''):
    mean_throttle = reference_df['AvgThrottle (%)'].mean()
    mean_brake = reference_df['AvgBrake (0-1)'].mean()

    simulated_stint = []
    for LapNumber in range(1, stint_length + 1):
        TyreAge = LapNumber
        FuelRemaining = max(0, start_fuel - (LapNumber - 1) * fuel_burn_per_lap)
        simulated_stint.append({
            'LapNumber': LapNumber,
            'TyreAge': TyreAge,
            'FuelRemaining': FuelRemaining,
            'AvgThrottle (%)': mean_throttle,
            'AvgBrake (0-1)': mean_brake
        })

    stint_df = pd.DataFrame(simulated_stint)
    features = ['TyreAge', 'FuelRemaining', 'AvgThrottle (%)', 'AvgBrake (0-1)']
    stint_df['PredictedLapTime'] = model.predict(stint_df[features])
    total_time = stint_df['PredictedLapTime'].sum()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(stint_df['LapNumber'], stint_df['PredictedLapTime'], marker='o', label='Predicted Lap Time')
    plt.xlabel('Lap Number')
    plt.ylabel('Predicted Lap Time (seconds)')
    plt.title(f'Predicted Lap Time Evolution - {title_suffix}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Total predicted stint time for {title_suffix}: {total_time:.2f} sec")
    return stint_df, total_time

soft_len = 20
medium_len = 20
hard_len = 20

soft_sim_df, soft_total = simulate_stint_and_plot(model_soft, soft_len, 110, 1.8, VER_LongestStint_SOFT_2025, title_suffix='SOFT - VER - Spanish GP 2025')
medium_sim_df, medium_total = simulate_stint_and_plot(model_medium, medium_len, 110, 1.8, VER_LongestStint_MEDIUM_2025, title_suffix='MEDIUM - VER - Spanish GP 2025')
hard_sim_df, hard_total = simulate_stint_and_plot(model_hard, hard_len, 110, 1.8, VER_LongestStint_HARD_2025, title_suffix='HARD - VER - Spanish GP 2025')

