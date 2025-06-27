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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import random

cache = '/Users/arthavpatel/Desktop/race_outcome_prediction/f1_cache'
f1.Cache.enable_cache(cache)

grand_prix = 'Spanish Grand Prix'
year = 2025
session_name = 'R'
session = f1.get_session(year,grand_prix, session_name)
session.load()
# ---------------------------------------------------------------------------------------------

# Assumations variables 
total_fuel = 110
fuel_burn_per_lap = 1.8

# Finding data 
laps = session.laps
VER_laps = laps[(laps['Driver'] == 'VER') & (laps['IsAccurate'] == True)]
VER_laps_SOFT = VER_laps[VER_laps['Compound'] == 'SOFT']
stint_start_lap = VER_laps_SOFT['LapNumber'].min()
fuel_at_start_stint = total_fuel - (stint_start_lap - 1) * fuel_burn_per_lap

# Avg Throttle and braking 
avg_throttle = []
avg_brake = []
for i in range(len(VER_laps_SOFT)):
    LapNumber = VER_laps_SOFT.iloc[i]['LapNumber']
    try:
        lap_data = VER_laps[VER_laps['LapNumber'] == LapNumber].iloc[0]
        car_data = lap_data.get_car_data()
        avg_throttle.append(car_data['Throttle'].mean())
        avg_brake.append(car_data['Brake'].mean())
    except Exception as e:
        avg_throttle.append(None)
        avg_brake.append(None)

VER_laps_SOFT = VER_laps_SOFT.copy()
VER_laps_SOFT['Driver'] = 'VER'
VER_laps_SOFT['Session'] = 'R'+'2025'
VER_laps_SOFT['FuelRemaining'] = fuel_at_start_stint - (VER_laps_SOFT['LapNumber'] - stint_start_lap) * fuel_burn_per_lap
VER_laps_SOFT['TyreAge'] = VER_laps_SOFT['LapNumber'] - stint_start_lap + 1
VER_laps_SOFT['LapTimeSeconds'] = VER_laps_SOFT['LapTime'].dt.total_seconds()
VER_laps_SOFT['Sector1Seconds'] = VER_laps_SOFT['Sector1Time'].dt.total_seconds().fillna(0)
VER_laps_SOFT['Sector2Seconds'] = VER_laps_SOFT['Sector2Time'].dt.total_seconds().fillna(0)
VER_laps_SOFT['Sector3Seconds'] = VER_laps_SOFT['Sector3Time'].dt.total_seconds().fillna(0)
VER_laps_SOFT['AVG_Throttle (%)'] = avg_throttle
VER_laps_SOFT['AVG_Braking (0-1)'] = avg_brake
base_line_time = VER_laps_SOFT['LapTimeSeconds'].iloc[0]
VER_laps_SOFT['DeltaTime'] = VER_laps_SOFT['LapTimeSeconds'] - base_line_time
cols = [
    'Driver', 'Session', 'LapNumber', 'LapTimeSeconds', 'Compound', 'FuelRemaining', 'TyreAge', 'TrackStatus',
    'Sector1Seconds', 'Sector2Seconds', 'Sector3Seconds', 'DeltaTime', 'AVG_Throttle (%)', 'AVG_Braking (0-1)'
]

VER_laps_SOFT[cols].to_csv("trail.csv", index=False)

soft_laps = VER_laps_SOFT['LapNumber'].tolist()
stint_starts = [soft_laps[0]]

for i in range(1, len(soft_laps)):
    if soft_laps[i] != soft_laps[i-1] + 1:
        stint_starts.append(soft_laps[i])

# Get VER's Medium tyre stints
VER_laps_MEDIUM = VER_laps[VER_laps['Compound'] == 'MEDIUM']
medium_laps = VER_laps_MEDIUM['LapNumber'].tolist()

# Identify stint start laps (based on non-consecutive gaps)
stint_starts_M = [medium_laps[0]]
for i in range(1, len(medium_laps)):
    if medium_laps[i] != medium_laps[i - 1] + 1:
        stint_starts_M.append(medium_laps[i])

# Calculate initial fuel for the first stint
stint_start_lap_M = VER_laps_MEDIUM['LapNumber'].min()
fuel_at_start_stint_M = total_fuel - (stint_start_lap_M - 1) * fuel_burn_per_lap

# Compute avg throttle and braking
avg_throttle_M = []
avg_brake_M = []
for i in range(len(VER_laps_MEDIUM)):
    LapNumber = VER_laps_MEDIUM.iloc[i]['LapNumber']
    try:
        lap_data = VER_laps[VER_laps['LapNumber'] == LapNumber].iloc[0]
        car_data = lap_data.get_car_data()
        avg_throttle_M.append(car_data['Throttle'].mean())
        avg_brake_M.append(car_data['Brake'].mean())
    except Exception:
        avg_throttle_M.append(None)
        avg_brake_M.append(None)

# Prepare columns
VER_laps_MEDIUM = VER_laps_MEDIUM.copy()
VER_laps_MEDIUM['Driver'] = 'VER'
VER_laps_MEDIUM['Session'] = 'R' + str(year)
VER_laps_MEDIUM['FuelRemaining'] = fuel_at_start_stint_M - (VER_laps_MEDIUM['LapNumber'] - stint_start_lap_M) * fuel_burn_per_lap
VER_laps_MEDIUM['TyreAge'] = VER_laps_MEDIUM['LapNumber'] - stint_start_lap_M + 1
VER_laps_MEDIUM['LapTimeSeconds'] = VER_laps_MEDIUM['LapTime'].dt.total_seconds()
VER_laps_MEDIUM['Sector1Seconds'] = VER_laps_MEDIUM['Sector1Time'].dt.total_seconds().fillna(0)
VER_laps_MEDIUM['Sector2Seconds'] = VER_laps_MEDIUM['Sector2Time'].dt.total_seconds().fillna(0)
VER_laps_MEDIUM['Sector3Seconds'] = VER_laps_MEDIUM['Sector3Time'].dt.total_seconds().fillna(0)
VER_laps_MEDIUM['AVG_Throttle (%)'] = avg_throttle_M
VER_laps_MEDIUM['AVG_Braking (0-1)'] = avg_brake_M
baseline_time_M = VER_laps_MEDIUM['LapTimeSeconds'].iloc[0]
VER_laps_MEDIUM['DeltaTime'] = VER_laps_MEDIUM['LapTimeSeconds'] - baseline_time_M

# Save to CSV
VER_laps_MEDIUM[cols].to_csv("trail_medium.csv", index=False)


# Finds the stints 
def extract_stints(df):
    df = df.sort_values('LapNumber')
    stints = []
    current_stint = [df.iloc[0]]

    for i in range(1, len(df)):
        if df.iloc[i]['LapNumber'] != df.iloc[i-1]['LapNumber'] + 1:
            stints.append(pd.DataFrame(current_stint))
            current_stint = []
        current_stint.append(df.iloc[i])

    if current_stint:
        stints.append(pd.DataFrame(current_stint))

    return stints

soft_stints = extract_stints(VER_laps_SOFT)
stint1 = soft_stints[0]

medium_stints = extract_stints(VER_laps_MEDIUM)
stint_1_medium = medium_stints[0]
# ---------------------------------------------------------------------------------------------
def train_ploy_model(stint_df, degree = 2):
    features = ['TyreAge', 'FuelRemaining', 'AVG_Throttle (%)', 'AVG_Braking (0-1)']
    X = stint_df[features].fillna(0)
    Y = stint_df['LapTimeSeconds']
    ploy_model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(2)),
        ('model', LinearRegression())
    ])
    ploy_model.fit(X,Y)
    return ploy_model

model_soft = train_ploy_model(stint1)
feature = ['TyreAge', 'FuelRemaining', 'AVG_Throttle (%)', 'AVG_Braking (0-1)']
X_SOFT = stint1[feature].fillna(0)
stint1['PredictedTimeSeconds'] = model_soft.predict(X_SOFT)

model_medium = train_ploy_model(stint_1_medium)
X_MEDIUM = stint_1_medium[feature].fillna(0)
stint_1_medium['PredictedTimeSeconds'] = model_medium.predict(X_MEDIUM)

plt.figure(figsize=(10, 6))
plt.plot(stint1['LapNumber'], stint1['LapTimeSeconds'], label='Actual')
plt.plot(stint1['LapNumber'], stint1['PredictedTimeSeconds'], label='Predicted', linestyle='--')
plt.xlabel('Lap Number')
plt.ylabel('Lap Time (s)')
plt.title('Actual vs Predicted Lap Time (SOFT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(stint_1_medium['LapNumber'], stint_1_medium['LapTimeSeconds'], label='Actual')
plt.plot(stint_1_medium['LapNumber'], stint_1_medium['PredictedTimeSeconds'], label='Predicted', linestyle='--')
plt.xlabel('Lap Number')
plt.ylabel('Lap Time (s)')
plt.title('Actual vs Predicted Lap Time (SOFT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------------------------------------------
def simulate_stint_and_plot(model, stint_length, start_fuel, fuel_burn_per_lap, reference_df, title_suffix=''):
    mean_throttle = reference_df['AVG_Throttle (%)'].mean()
    mean_brake = reference_df['AVG_Braking (0-1)'].mean()

    simulated_stint = []
    for LapNumber in range(1, stint_length + 1):
        TyreAge = LapNumber
        FuelRemaining = max(0, start_fuel - (LapNumber - 1) * fuel_burn_per_lap)
        simulated_stint.append({
            'LapNumber': LapNumber,
            'TyreAge': TyreAge,
            'FuelRemaining': FuelRemaining,
            'AVG_Throttle (%)': mean_throttle,
            'AVG_Braking (0-1)': mean_brake
        })

    stint_df = pd.DataFrame(simulated_stint)
    features = ['TyreAge', 'FuelRemaining', 'AVG_Throttle (%)', 'AVG_Braking (0-1)']
    stint_df['PredictedTimeSeconds'] = model.predict(stint_df[features])
    total_time = stint_df['PredictedTimeSeconds'].sum()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(stint_df['LapNumber'], stint_df['PredictedTimeSeconds'], marker='o', label='Predicted Lap Time')
    plt.xlabel('Lap Number')
    plt.ylabel('Predicted Lap Time (seconds)')
    plt.title(f'Predicted Lap Time Evolution - {title_suffix}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Total predicted stint time for {title_suffix}: {total_time:.2f} sec")
    return stint_df, total_time

soft_len = 4
soft_sim_df, soft_total = simulate_stint_and_plot(model_soft, soft_len, 110, 1.8, stint1, title_suffix='SOFT - VER - Spanish GP 2025')

medium_len = 4
medium_sim_df, medium_total = simulate_stint_and_plot(model_medium, medium_len, 110, 1.8, stint_1_medium, title_suffix='MEDIUM - VER - Spanish GP 2025') 