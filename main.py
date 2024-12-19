import pandas as pd
from pathlib import Path
from pyulog import ULog
import matplotlib.pyplot as plt
from metpy.plots import SkewT, Hodograph
from metpy.units import units
import numpy as np
import logging
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_ulog_to_dataframe(ulog_path, selected_columns):
    try:
        # logging.info(f"Loading ULog file: {ulog_path}")
        ulog = ULog(ulog_path)

        master_df = pd.DataFrame()

        for topic, columns in selected_columns.items():
            try:
                message = next(msg for msg in ulog.data_list if msg.name == topic)
                df = pd.DataFrame(message.data)

                df = df[["timestamp"] + columns]
                df["timestamp"] = df["timestamp"] / 1e6  # Convert timestamp to seconds

                df.set_index("timestamp", inplace=True)

                df.index = pd.to_datetime(df.index, unit='s').round('s')

                df_resampled = df.resample('1s').mean()

                if master_df.empty:
                    master_df = df_resampled
                else:
                    master_df = master_df.join(df_resampled, how='outer')
            except StopIteration:
                logging.warning(f"Topic {topic} not found in {ulog_path}")

        master_df.dropna(how="all", inplace=True)
        return master_df
    except Exception as e:
        logging.error(f"Error processing ULog file: {e}")
        raise

def calculate_derived_values(df):
    try:
        # Calculate wind direction and magnitude
        # logging.info("Calculating wind direction and magnitude")
        df['wind_magnitude'] = np.sqrt(df['windspeed_north']**2 + df['windspeed_east']**2)
        df['wind_direction'] = np.arctan2(df['windspeed_east'], df['windspeed_north']) * (180 / np.pi)

        # Calculate dewpoint (using WMO-recommended formula)
        RH = df['sht_humidity'] / 100.0
        T = df['therm_temp_celcius']

        # Compute actual vapor pressure using the WMO-recommended constants (17.67 and 243.5 °C)
        e = 6.112 * np.exp((17.67 * T) / (T + 243.5)) * RH

        # Solve the inverse to find dew point
        df['dewpoint'] = (243.5 * np.log(e / 6.112)) / (17.67 - np.log(e / 6.112))
        return df
    except Exception as e:
        logging.error(f"Error calculating derived values: {e}")
        raise

def get_user_index_range(df):
    try:

        print("\nDataFrame index format example:", df.index[0].strftime("%Y-%m-%d %H:%M:%S"))
        print("\nEnter time range for plotting (format: HH:MM:SS)")
        print("\nUse the time range when the drone was ascending. Holding at altitude causes icing and unreliable temperature data on the way back down")

        first_time = df.index[0].strftime("%H:%M:%S")
        last_time = df.index[-1].strftime("%H:%M:%S") 

        print(f"Available time range: {first_time} to {last_time}")
        start_time = input("Enter start time (HH:MM:SS): ")
        end_time = input("Enter end time (HH:MM:SS): ")

        base_date = df.index[0].strftime("%Y-%m-%d")
        start_datetime = pd.to_datetime(f"{base_date} {start_time}")
        end_datetime = pd.to_datetime(f"{base_date} {end_time}")

        return start_datetime, end_datetime
    except Exception as e:
        logging.error(f"Error getting user index range: {e}")
        raise

def plot_skewt(df):
    try:
        # logging.info("Preparing data for SkewT plot").
        pressure = df['pressure'].values * units.Pa
        temperature = df['therm_temp_celcius'].values * units.degC
        dewpoint = df['dewpoint'].values * units.degC
        u_wind = df['windspeed_north'].values * units.meter_per_second
        v_wind = df['windspeed_east'].values * units.meter_per_second
        wind_speed = df['wind_magnitude'].values * units.meter_per_second

        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, rotation=45)
        
        skew.ax.grid(True)
        
        skew.ax.set_ylim(1000, 500)
        skew.ax.set_xlim(-15, 25)

        skew.plot(pressure, temperature,'r', lw=2, label='TEMPERATURE')
        skew.plot(pressure, dewpoint,'g', lw=2, label='DEWPOINT')
        
        # Sample every nth point to get 10 total barbs
        n = len(pressure) // 10
        if n < 1: n = 1  # Handle case with less than 10 points
        
        skew.plot_barbs(pressure[::n], u_wind[::n], v_wind[::n])

        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()
        
        skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
        
        ax_hod = inset_axes(skew.ax, '40%', '40%', loc=1)
        h = Hodograph(ax_hod, component_range=wind_speed.max().magnitude*1.2)
        h.add_grid(increment=5)
        h.plot_colormapped(u_wind, v_wind, wind_speed)  
        # Add legend and labels
        skewleg = skew.ax.legend(loc='upper left')
        plt.title(f"Skew-T Log-P for {Path(ulog_path).name}")
        plt.show()
    except Exception as e:
        logging.error(f"Error creating SkewT plot: {e}")
        raise

# Example Usage
if __name__ == "__main__":
    try:
        ulog_dir = Path(__file__).parent / "ulog"
        ulog_files = list(ulog_dir.glob("*.ulg"))
        print("Available .ulg files:")
        for f in ulog_files:
            print(f"  {f.name}")
        if len(ulog_files) == 0:
            print("No .ulg files found in the ulog directory.")
            exit()
        ulog_path = str(ulog_files[0])

        selected_columns = {
            'sensor_baro': ['pressure'],
            'todd_sensor': ['sht_humidity', 'therm_temp_celcius'],
            'wind': ['windspeed_north', 'windspeed_east']
        }

        df = process_ulog_to_dataframe(ulog_path, selected_columns)

        df = calculate_derived_values(df)

        # Get user-specified index range
        start_time, end_time = get_user_index_range(df)
        df = df[(df.index >= start_time) & (df.index <= end_time)]
                
        # Plot the SkewT diagram
        plot_skewt(df)
    except Exception as e:
        logging.critical(f"Critical error in main program: {e}")
