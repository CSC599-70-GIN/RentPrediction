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
#Extracting important observations from the features
df = results_df[['cmplnt_num', 'rpt_dt', 'addr_pct_cd', 'ofns_desc', 'law_cat_cd']]
df = df.dropna()
df = df.loc[
    (df.ofns_desc == 'PETIT LARCENY') 
    | (df.ofns_desc == 'HARRASSMENT 2')
    | (df.ofns_desc == 'ASSAULT 3 & RELATED OFFENSES')
    | (df.ofns_desc == 'CRIMINAL MISCHIEF & RELATED OF')
    | (df.ofns_desc == 'GRAND LARCENY')
    | (df.ofns_desc == 'OFF. AGNST PUB ORD SENSBLTY &')
    | (df.ofns_desc == 'FELONY ASSAULT')
    | (df.ofns_desc == 'DANGEROUS DRUGS')
    | (df.ofns_desc == 'MISCELLANEOUS PENAL LAW')
    | (df.ofns_desc == 'ROBBERY')
    | (df.ofns_desc == 'BURGLARY')
]
#Converting the observations to a form that we can use in our model
df['petit_larceny'] = (df.ofns_desc == 'PETIT LARCENY')
df['harrassment_2'] = (df.ofns_desc == 'HARRASSMENT 2')
df['assault_3'] = (df.ofns_desc == 'ASSAULT 3 & RELATED OFFENSES')
df['criminal_mischief'] = (df.ofns_desc == 'CRIMINAL MISCHIEF & RELATED OF')
df['grand_larceny'] = (df.ofns_desc == 'GRAND LARCENY')
df['public_order'] = (df.ofns_desc == 'OFF. AGNST PUB ORD SENSBLTY &')
df['felony_assault'] = (df.ofns_desc == 'FELONY ASSAULT')
df['dangerous_drugs'] = (df.ofns_desc == 'DANGEROUS DRUGS')
df['penal_law'] = (df.ofns_desc == 'MISCELLANEOUS PENAL LAW')
df['robbery'] = (df.ofns_desc == 'ROBBERY')
df['burglary'] = (df.ofns_desc == 'BURGLARY')
df['misdemenor'] = (df.law_cat_cd == 'MISDEMEANOR')
df['felony'] = (df.law_cat_cd == 'FELONY')
df['violation'] = (df.law_cat_cd == 'VIOLATION')

df['addr_pct_cd'] = df['addr_pct_cd'].astype(int)

df = df[
   ['addr_pct_cd', 'petit_larceny', 'harrassment_2', 'assault_3', 'criminal_mischief', 'grand_larceny', 'public_order', 'felony_assault', 'dangerous_drugs', 'penal_law', 'robbery', 'burglary', 'misdemenor', 'felony', 'violation']
   ].groupby(['addr_pct_cd'], as_index=False).sum()

print(df.sample(10), df.shape)

df.to_csv('2018_NYPD_Complaints.csv', index=False)