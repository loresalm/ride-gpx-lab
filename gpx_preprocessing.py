import gpxpy
import pandas as pd
import numpy as np
from haversine import haversine, Unit
from datetime import datetime
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# --- Configuration ---
GPX_FILE = "data/2025-06-20T06-39-02.036Z_Cycling.gpx"
OUTPUT_CSV = "ride_data.csv"
OUTPUT_PNG = "velocity_comparison.png"

def parse_gpx(gpx_file):
    """Parse GPX file and ensure proper datetime conversion"""
    with open(gpx_file, "r") as f:
        gpx = gpxpy.parse(f)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    "lat": point.latitude,
                    "lon": point.longitude,
                    "time": pd.to_datetime(point.time),  # Convert immediately
                })
    return gpx, pd.DataFrame(points)

def ensure_datetime(df):
    """Ensure time column is proper datetime"""
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    return df

def calculate_velocity(df):
    """Calculate velocity between points"""
    df = ensure_datetime(df)
    df = df.sort_values('time').reset_index(drop=True)
    
    # Calculate time differences in hours
    time_diff = df['time'].diff().dt.total_seconds() / 3600
    
    # Calculate distances between points in km
    distances = [np.nan]
    for i in range(1, len(df)):
        point1 = (df.at[i-1, 'lat'], df.at[i-1, 'lon'])
        point2 = (df.at[i, 'lat'], df.at[i, 'lon'])
        distances.append(haversine(point1, point2, unit=Unit.KILOMETERS))
    
    # Calculate velocity in km/h
    df['vel_kmh'] = np.array(distances) / time_diff
    return df

def filter_velocity(df, min_speed, max_speed): 
    """Filter out low-speed points"""
    # Start with all valid (non-NA) points
    mask = ~df['vel_kmh'].isna()
    
    # Apply minimum speed threshold if provided
    if min_speed is not None:
        mask = mask & (df['vel_kmh'] >= min_speed)
    
    # Apply maximum speed threshold if provided
    if max_speed is not None:
        mask = mask & (df['vel_kmh'] <= max_speed)
    
    return df[mask]

def smooth_and_downsample(df, smooth_window=5, max_points=1000, min_distance=10):
    """Smooth and downsample while preserving velocity profile"""
    df = ensure_datetime(df)
    
    # Smoothing
    if len(df) >= smooth_window:
        df['vel_kmh'] = savgol_filter(
            df['vel_kmh'].fillna(0),
            window_length=smooth_window,
            polyorder=2,
            mode='nearest'
        )
    
    # Downsampling
    coords = df[['lat', 'lon']].values
    distances = [0]
    for i in range(1, len(coords)):
        distances.append(haversine(coords[i-1], coords[i], unit='m'))
    
    df['cum_dist_m'] = np.cumsum(distances)
    total_dist = df['cum_dist_m'].iloc[-1]
    target_interval = max(min_distance, total_dist/max_points)
    
    keep_indices = [0]
    last_kept_dist = 0
    for i in range(1, len(df)):
        if (df['cum_dist_m'].iloc[i] - last_kept_dist) >= target_interval:
            keep_indices.append(i)
            last_kept_dist = df['cum_dist_m'].iloc[i]
    
    if len(df) - 1 not in keep_indices:
        keep_indices.append(len(df) - 1)
    
    return df.iloc[keep_indices].reset_index(drop=True)

def create_velocity_plots(original_df, filtered_df, smoothed_df, output_png):
    """Create comparison plot of velocity profiles"""
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Original Velocity
    plt.subplot(3, 1, 1)
    plt.plot(original_df['vel_kmh'], 'b-', alpha=0.7)
    plt.title('Original Velocity')
    plt.ylabel('Velocity (km/h)')
    plt.grid(True)
    
    # Subplot 2: After Threshold Filtering
    plt.subplot(3, 1, 2)
    plt.plot(filtered_df['vel_kmh'], 'g-', alpha=0.7)
    plt.title(f'After Filtering km/h)')
    plt.ylabel('Velocity (km/h)')
    plt.grid(True)
    
    # Subplot 3: After Smoothing and Downsampling
    plt.subplot(3, 1, 3)
    plt.plot(smoothed_df['vel_kmh'], 'r-', alpha=0.7)
    plt.title('After Smoothing and Downsampling')
    plt.ylabel('Velocity (km/h)')
    plt.xlabel('Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Velocity comparison plot saved to {output_png}")

# --- Main Processing Pipeline ---
minspeed = 1  # km/h
maxspeed = 70  # km/h

# Step 1: Parse and convert timestamps
gpx, df = parse_gpx(GPX_FILE)

# Step 2: Calculate velocity
df = calculate_velocity(df)
original_vel_df = df.copy()
print(f"original df size: {len(df)}")

# Step 3: Filter velocity
filtered_df = filter_velocity(df.copy(), minspeed, maxspeed)
print(f"min max filtered df size: {len(filtered_df)}")

# Step 4: Smooth and downsample
smoothed_df = smooth_and_downsample(filtered_df.copy())
print(f"smoothing and downsample df size: {len(smoothed_df)}")

# Create plots and save output
create_velocity_plots(original_vel_df, filtered_df, smoothed_df, OUTPUT_PNG)
smoothed_df.to_csv(OUTPUT_CSV, index=False)