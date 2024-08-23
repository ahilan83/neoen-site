import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pvlib import location
from pvlib import irradiance
from timezonefinder import TimezoneFinder
from pytz import timezone
from mpl_toolkits.mplot3d import Axes3D
import pvlib



Calculate_POA(ghi,dni,dhi,albedo,lat,lon,azimuth_angle,tilt,tart_date=start_date,end_date=end_date):

    # Find the timezone name using TimezoneFinder
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)

    # Use pytz to get the timezone object
    tz = timezone(tz_name)

    # Create location object to store lat, lon, timezone
    site_location = location.Location(lat, lon, tz=tz)

    times = pd.date_range(start=start_date, end=end_date, freq='H')
    solar_position = site_location.get_solarposition(times=times)

    # Extract zenith and azimuth angles
    zenith_angles = solar_position['zenith']
    azimuth_angles = solar_position['azimuth']

    aoi = calculate_dynamic_aoi(tilt, azimuth_angle, solar_position['azimuth'], 90-solar_position['zenith'])
    POA = calculate_poa_irradiance(dni, dhi, ghi, aoi, tilt, albedo)
    return POA



def calculate_dynamic_aoi(tilt, azimuth_angle, solar_azimuth, solar_zenith):
    # Convert angles from degrees to radians
    azimuth_angle_rad = np.radians(azimuth_angle)
    solar_azimuth_rad = np.radians(solar_azimuth)
    solar_zenith_rad = np.radians(solar_zenith)
    
    # For single-axis tracker, tilt is approximately the solar zenith angle
    tilt_rad = np.radians(tilt)
    
    # Calculate AOI
    aoi = np.degrees(np.arccos(
        np.cos(solar_zenith_rad) * np.cos(tilt_rad) +
        np.sin(solar_zenith_rad) * np.sin(tilt_rad) * np.cos(solar_azimuth_rad - azimuth_angle_rad)
 ))
    
    return aoi


def calculate_poa_irradiance(dni, dhi, ghi, aoi, tilt, albedo):
    """
    Calculate the plane of array (POA) irradiance.
    
    Parameters:
    dni (float): Direct Normal Irradiance (W/m^2)
    dhi (float): Diffuse Horizontal Irradiance (W/m^2)
    ghi (float): Global Horizontal Irradiance (W/m^2)
    aoi (float): Angle of Incidence (degrees)
    tilt (float): Tilt angle of the panel (degrees)
    albedo (float): Ground reflectance (default is 0.2)
    
    Returns:
    float: Total POA irradiance (W/m^2)
    """   
    # Convert angles from degrees to radians
    aoi_rad = np.radians(aoi)
    tilt_rad = np.radians(tilt)
    
    # Direct component
    poa_direct = dni * np.cos(aoi_rad)

    # Diffuse component (using the isotropic sky model)
    poa_diffuse = dhi * (1 + np.cos(tilt_rad)) / 2

    # Ground-reflected component
    poa_ground_reflected = ghi * albedo * (1 - np.cos(tilt_rad)) / 2

    # Total POA irradiance
    poa_total = poa_direct + poa_diffuse + poa_ground_reflected
    
    return poa_total

