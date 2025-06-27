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
import random
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
    # plt.show()

    print(f"Total predicted stint time for {title_suffix}: {total_time:.2f} sec")
    return stint_df, total_time

soft_len = 20
medium_len = 20
hard_len = 20

soft_sim_df, soft_total = simulate_stint_and_plot(model_soft, soft_len, 110, 1.8, VER_LongestStint_SOFT_2025, title_suffix='SOFT - VER - Spanish GP 2025')
medium_sim_df, medium_total = simulate_stint_and_plot(model_medium, medium_len, 110, 1.8, VER_LongestStint_MEDIUM_2025, title_suffix='MEDIUM - VER - Spanish GP 2025')
hard_sim_df, hard_total = simulate_stint_and_plot(model_hard, hard_len, 110, 1.8, VER_LongestStint_HARD_2025, title_suffix='HARD - VER - Spanish GP 2025')

# ---------------------------------- Phase 5 ------------------------------------ #
fuel_penalty = 0.035  # sec per kg of fuel

soft_df = VER_LongestStint_SOFT_2025.copy()
soft_df['AdjLapTime'] = soft_df['LapTimeSeconds'] - soft_df['FuelRemaining'] * fuel_penalty

medium_df = VER_LongestStint_MEDIUM_2025.copy()
medium_df['AdjLapTime'] = medium_df['LapTimeSeconds'] - medium_df['FuelRemaining'] * fuel_penalty

hard_df = VER_LongestStint_HARD_2025.copy()
hard_df['AdjLapTime'] = hard_df['LapTimeSeconds'] - hard_df['FuelRemaining'] * fuel_penalty

def train_degradation_model(stint_df, degree = 2):
    X = stint_df[['TyreAge']]
    y = stint_df['AdjLapTime']
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X,y)
    return model

deg_model_soft = train_degradation_model(soft_df)
deg_model_medium = train_degradation_model(medium_df)
deg_model_hard = train_degradation_model(hard_df)


def simulate_degradation(model, stint_length, compound):
    laps = pd.DataFrame({'TyreAge': range(1,stint_length+1)})
    laps['PredictedLapTime'] = model.predict(laps[['TyreAge']])
    laps['Compound'] = compound
    return laps

soft_deg = simulate_degradation(deg_model_soft, 66, 'SOFT')
medium_deg = simulate_degradation(deg_model_medium, 66, 'MEDIUM')
hard_deg = simulate_degradation(deg_model_hard, 66, 'HARD')

together = pd.concat([soft_deg, medium_deg, hard_deg])
plt.figure(figsize=(10,6))
for compound in together['Compound'].unique():
    subset = together[together['Compound'] == compound]
    plt.plot(subset['TyreAge'], subset['PredictedLapTime'], label = compound)
plt.xlabel("Tyre Age (laps)")
plt.ylabel("Predicted Lap Time (s)")
plt.title("Tyre Degradation Curve (Fuel-Independent)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

pivoted = together.pivot(index='TyreAge', columns='Compound', values='PredictedLapTime')
pivoted['SOFT - MEDIUM'] = pivoted['SOFT'] - pivoted['MEDIUM']
pivoted['SOFT - HARD'] = pivoted['SOFT'] - pivoted['HARD']

pivoted[['SOFT - MEDIUM', 'SOFT - HARD']].plot(title="Lap Time Delta vs SOFT", grid=True)
plt.xlabel("Tyre Age")
plt.ylabel("Lap Time Delta (s)")
plt.tight_layout()
# plt.show()

# ---------------------------------- Phase 6 ------------------------------------ #
simulations = 10
race_length = 66

clean_laps = VER_Laps_R_2025[
    (VER_Laps_R_2025['Driver'] == 'VER') &
    (VER_Laps_R_2025['IsAccurate'] == True) &
    (VER_Laps_R_2025['PitInTime'].isna()) & 
    (VER_Laps_R_2025['PitOutTime'].isna()) &
    (VER_Laps_R_2025['Deleted'] == False)    
].copy()

clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()
avg_time_seconds = clean_laps['LapTimeSeconds'].mean()

# Avg Pit Stop Time
VER_Laps = VER_Laps_R_2025[VER_Laps_R_2025['Driver'] == 'VER'].copy()
VER_Laps = VER_Laps[['LapNumber', 'PitInTime', 'PitOutTime']].reset_index(drop=True)
pit_stops = []
for i in range(len(VER_Laps) - 1):
    in_time = VER_Laps.loc[i, 'PitInTime']
    out_time = VER_Laps.loc[i+1, 'PitOutTime']

    if pd.notna(in_time) and pd.notna(out_time):
        duration = (out_time - in_time).total_seconds()
        lap = VER_Laps.loc[i, 'LapNumber']
        pit_stops.append({
            'Lap' : lap,
            'PitInTime' : in_time,
            'PitOutTime' : out_time,
            'PitDuration' : duration
        })
pit_df = pd.DataFrame(pit_stops)
avg_pit_time = pit_df['PitDuration'].mean()

# Degradations 
def calculate_degradation(compound, compound_name, fallback):
   degradations = []
  
   if compound.empty:
       print(f"No {compound_name} Tyres were used in this GRAND PRIX")
       return fallback


   compound_stints = compound.groupby('Stint')
   for stint_num, stint in compound_stints:
       stint = stint.sort_values('LapNumber')
       if len(stint) < 3:
           continue
       first = stint['LapTimeSeconds'].iloc[0]
       last = stint['LapTimeSeconds'].iloc[-1]
       degradation = (last - first) / (len(stint) - 1)
       degradations.append(degradation)


   return np.mean(degradations) if degradations else None
default_soft_deg = 0.10
default_medium_deg = 0.05
default_hard_deg = 0.02


soft_stint = clean_laps[clean_laps['Compound'] == 'SOFT']
medium_stint = clean_laps[clean_laps['Compound'] == 'MEDIUM']
hard_stint = clean_laps[clean_laps['Compound'] == 'HARD']


avg_soft_degradation = calculate_degradation(soft_stint, "Soft", default_soft_deg)
avg_medium_degradation = calculate_degradation(medium_stint, "Medium", default_medium_deg)
avg_hard_degradation = calculate_degradation(hard_stint, "Hard", default_hard_deg)

# Safety Car deployment 
year = range(2017, 2025)
total_safety_cars = 0
total_virtual_safety_car = 0
total_red_flags = 0
total_races = len(years)
for i in years:
   try:
       session = f1.get_session(i, grand_prix_name, 'R')
       session.load()
       track_status = session.track_status
       track_status['Status'] = track_status['Status'].astype(str)

       unique_flags = track_status['Status'].unique()
       is_red_flag = '6' in unique_flags
       is_sc = '4' in unique_flags
       is_vsc = '5' in unique_flags

       if is_sc and not is_red_flag:
           total_safety_cars += 1 
       if is_sc and is_red_flag:
           total_red_flags += 1        
       if is_vsc:
           total_virtual_safety_car += 1
   except:
       print(f"Failed to load {i}")

total_prob_SC = total_safety_cars / total_races
total_prob_VSC = total_virtual_safety_car / total_races
total_prob_red_flag = total_red_flags / total_races

# Driver Profile 
print("--------------------------------------------------------------------")
print(f"Total Simulations: {simulations}")
print(f"Total laps in {grand_prix_name}: {race_length}")
print(f"Average lap time of Max Verstappen: {avg_time_seconds:.3f} sec")
print(f"Average pit time: {avg_pit_time:.1f} sec")
print("--------------------------------------------------------------------")
print("Average degradations of compounds")
print(f"Soft Tyres: {avg_soft_degradation:.3f} sec/lap")
print(f"Medium Tyres: {avg_medium_degradation:.3f} sec/lap")
print(f"Hard Tyres: {avg_hard_degradation:.3f} sec/lap")
print("--------------------------------------------------------------------")
print(f"Total Safety Car Probability: {total_prob_SC:.1%}")
print(f"Total Virtual Safety Car Probability: {total_prob_VSC:.1%}")
print(f"Total Red Flag Probability: {total_prob_red_flag:.1%}")

class CarDriverProfile:
   def __init__(self, name, base_lap_time, avg_pit_time, soft_deg,
                medium_deg, hard_deg, variability_range,
                safety_car_chance, virtual_safety_car_chance,
                race_length,
                safety_car_laps=(2, 5), sc_speed_factor=0.75,
                virtual_safety_car_laps=(1, 3), vsc_speed_factor=0.75,
                fuel_burn_rate=1.7, fuel_penalty_per_kg=0.035,
                red_flag_chance=total_prob_red_flag, red_flag_time_saving=20):
      
       self.name = name
       self.base_lap_time = base_lap_time
       self.avg_pit_time = avg_pit_time
       self.soft_deg = soft_deg
       self.medium_deg = medium_deg
       self.hard_deg = hard_deg
       self.variability_range = variability_range


       self.safety_car_chance = safety_car_chance
       self.safety_car_laps = safety_car_laps
       self.sc_speed_factor = sc_speed_factor


       self.virtual_safety_car_chance = virtual_safety_car_chance
       self.virtual_safety_car_laps = virtual_safety_car_laps
       self.vsc_speed_factor = vsc_speed_factor


       self.race_length = race_length


       self.fuel_burn_rate = fuel_burn_rate
       self.fuel_penalty_per_kg = fuel_penalty_per_kg


       self.red_flag_chance = red_flag_chance
       self.red_flag_time_saving = red_flag_time_saving

verstappen = CarDriverProfile(
   name="Max Verstappen",
   base_lap_time=avg_time_seconds,
   avg_pit_time=avg_pit_time,
   soft_deg=avg_soft_degradation,
   medium_deg=avg_medium_degradation,
   hard_deg=avg_hard_degradation,
   variability_range=(-0.1, 0.1),
   safety_car_chance=total_prob_SC,
   virtual_safety_car_chance=total_prob_VSC,
   race_length=race_length,
   vsc_speed_factor=(0.65, 0.75) ,
   fuel_burn_rate=1.7,
   fuel_penalty_per_kg=0.035,
   red_flag_chance=total_prob_red_flag,       
   red_flag_time_saving=15    
)

# Simulate the race 
def simulate_race(driver_profile, tire_stints):
    total_time = 0
    laps = 0
    fuel_left = driver_profile.fuel_burn_rate * driver_profile.race_length

    # Safety car, VSC, red flag setups
    sc_deployed = random.random() < driver_profile.safety_car_chance
    sc_laps = random.randint(*driver_profile.safety_car_laps) if sc_deployed else 0
    sc_start = random.randint(10, driver_profile.race_length - sc_laps - 5) if sc_deployed else -1

    vsc_deployed = random.random() < driver_profile.virtual_safety_car_chance
    vsc_laps = random.randint(*driver_profile.virtual_safety_car_laps) if vsc_deployed else 0
    vsc_start = random.randint(10, driver_profile.race_length - vsc_laps - 3) if vsc_deployed else -1

    red_flag_deployed = random.random() < driver_profile.red_flag_chance
    red_flag_lap = random.randint(10, driver_profile.race_length - 10) if red_flag_deployed else -1
    red_flag_applied = False

    for i, (compound, stint_length) in enumerate(tire_stints):
        # Select appropriate model
        if compound.upper() == "SOFT":
            deg_model = deg_model_soft
        elif compound.upper() == "MEDIUM":
            deg_model = deg_model_medium
        else:
            deg_model = deg_model_hard

        for stint_lap in range(stint_length):
            laps += 1
            if laps > driver_profile.race_length:
                break

            tyre_age_df = pd.DataFrame({'TyreAge': [stint_lap + 1]})
            base_lap_time = deg_model.predict(tyre_age_df)[0]

            fuel_left -= driver_profile.fuel_burn_rate
            fuel_penalty = fuel_left * driver_profile.fuel_penalty_per_kg
            lap_time = base_lap_time + random.uniform(*driver_profile.variability_range) + fuel_penalty

            # SC, VSC, RF adjustment
            if sc_deployed and sc_start <= laps <= sc_start + sc_laps:
                lap_time /= driver_profile.sc_speed_factor
            elif vsc_deployed and vsc_start <= laps <= vsc_start + vsc_laps:
                lap_time /= random.uniform(*driver_profile.vsc_speed_factor)

            if red_flag_deployed and not red_flag_applied and laps == red_flag_lap:
                lap_time -= driver_profile.red_flag_time_saving
                red_flag_applied = True

            total_time += lap_time

        # Add pit stop unless last stint
        if i < len(tire_stints) - 1:
            pit_time = driver_profile.avg_pit_time
            if compound.upper() == "SOFT":
                pit_window = (15, 25)
            elif compound.upper() == "MEDIUM":
                pit_window = (25, 40)
            else:
                pit_window = (0, 0)

            if sc_deployed and pit_window[0] <= sc_start <= pit_window[1]:
                pit_time -= 8

            total_time += pit_time

    return float(total_time)

def monte_carlo_simulation(driver_profile, tire_stint, simulation=simulations):
   return [simulate_race(driver_profile, tire_stint) for _ in range(simulation)]

strategy_pool = [
   [("Soft", 10), ("Medium", 30), ("Medium", 26)],
   [("Medium", 22), ("Soft", 22), ("Medium", 22)],
   [("Medium", 20), ("Medium", 23), ("Soft", 23)],
   [("Hard", 20), ("Medium", 28), ("Soft", 18)],
   [("Soft", 22), ("Hard", 22), ("Soft", 22)],
   [("Medium", 18), ("Hard", 30), ("Soft", 18)],
   [("Medium", 18), ("Hard", 30), ("Medium", 18)],
]

strategy_results = []
for strategy in strategy_pool:
   result = monte_carlo_simulation(verstappen, strategy)
   avg_time = float(np.mean(result))
   strategy_results.append((strategy, avg_time))


strategy_results.sort(key=lambda x: x[1])
best_strategy, best_time = strategy_results[0]

print("\nBest 2-Stop Strategy:")
for stint in best_strategy:
   print(f"{stint[0]} for {stint[1]} laps")
print(f"Average Race Time: {best_time:.2f} seconds")

for i in range(len(strategy_results)):
    print(strategy_results[i])

