import pandas as pd
import reverse_geocoder as rg
from pprint import pprint
import certifi
import ssl
import geopy.geocoders

ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx

airbnb_df = pd.read_csv("AB_NYC_2019.csv") # external dataset

geolocator = geopy.geocoders.Nominatim()

def combine(lat, lon):
    coord = str(lat) + ',' + str(lon)
    return coord

def lookupAddress(lat,lon):
    location = str(geolocator.reverse(combine(lat, lon)))
    return location

# This times out because there are in fact too many requests being sent to the online API.
# There is also the problem of accessing the correct neighborhood from the address as it is either at the 3rd or 4th
# index.
airbnb_df['coords'] = airbnb_df.apply (lambda row: lookupAddress(row.latitude, row.longitude), axis=1)
