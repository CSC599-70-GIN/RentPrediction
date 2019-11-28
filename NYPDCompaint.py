import pandas as pd
from sodapy import Socrata

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
client = Socrata("data.cityofnewyork.us", None)

# Example authenticated client (needed for non-public datasets):
# client = Socrata(data.cityofnewyork.us,
#                  MyAppToken,
#                  userame="user@example.com",
#                  password="AFakePassword")

results = client.get("qgea-i56i", where="date_extract_y(rpt_dt) = 2018", limit=1000000)

# Convert to pandas DataFrame
results_df = pd.DataFrame.from_records(results)

print(results_df.sample(10), results_df.shape)

results_df.to_csv('2018_NYPD_Complaints.csv')