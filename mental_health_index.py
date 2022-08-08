# Goal of this script is to create a menthal health index that we can use as a measure in the risk metric
# Username: pshroff 
# Date: 02-17-2022

# installing packages 
import numpy as np 
import pandas as pd 
import re 


# Reading all the files containing the mental health data 
filePath = r"data/fmddata/"
fileNames = ["countyFMD_data_2016x.csv", "countyFMD_data_2017x.csv", "countyFMD_data_2018x.csv", "countyFMD_data_2020x.csv", "countyFMD_data_2021x.csv"]

df_14 = pd.read_csv(filePath+fileNames[0])
df_15 = pd.read_csv(filePath+fileNames[1])
df_16 = pd.read_csv(filePath+fileNames[2])
df_17 = pd.read_csv(filePath+fileNames[3])
df_18 = pd.read_csv(filePath+fileNames[4])

# Combining the 5 datasets into one big dataframe 
df = pd.concat([df_14,df_15,df_16,df_17, df_18], ignore_index=True)


def state_abbr(state):

	us_state_to_abbrev = {"Alabama": "AL","Alaska": "AK","Arizona": "AZ","Arkansas": "AR","California": "CA","Colorado": "CO","Connecticut": "CT","Delaware": "DE","Florida": "FL","Georgia": "GA","Hawaii": "HI",
	    "Idaho": "ID","Illinois": "IL","Indiana": "IN","Iowa": "IA","Kansas": "KS","Kentucky": "KY","Louisiana": "LA","Maine": "ME","Maryland": "MD","Massachusetts": "MA","Michigan": "MI","Minnesota": "MN","Mississippi": "MS",
	    "Missouri": "MO","Montana": "MT","Nebraska": "NE","Nevada": "NV","New-Hampshire": "NH","New Hampshire": "NH", "New-Jersey": "NJ", "New Jersey": "NJ", "New-Mexico": "NM", "New Mexico": "NM", "New-York": "NY",
	    "New York": "NY", "North-Carolina": "NC", "North Carolina": "NC", "North-Dakota": "ND", "North Dakota": "ND", "Ohio": "OH","Oklahoma": "OK", "Oregon": "OR","Pennsylvania": "PA","Rhode-Island": "RI", "Rhode Island": "RI",
	    "South-Carolina": "SC", "South Carolina": "SC", "South-Dakota": "SD", "South Dakota": "SD", "Tennessee": "TN","Texas": "TX","Utah": "UT","Vermont": "VT","Virginia": "VA","Washington": "WA","West-Virginia": "WV",
	    "West Virginia": "WV", "Wisconsin": "WI","Wyoming": "WY","District-Of-Columbia": "DC","District of Columbia": "DC", "American-Samoa": "AS", "American Samoa": "AS", "Guam": "GU","Northern-Mariana-Islands": "MP",
	    "Northtern Mariana Islands": "MP", "Puerto-Rico": "PR", "Puerto Rico": "PR", "United-States-Minor-Outlying-Islands": "UM", "United State Minor Outlying Islands": "UM", "U.S.-Virgin-Islands": "VI", 
	    "U.S. Virgin Islands": "VI"
	}

	return us_state_to_abbrev[state]

def update_year(year):
	year_change = {2016:2014, 2017:2015, 2018:2016, 2020:2017, 2021:2018}

	return year_change[year]

def extract_county(county, state, df_name):

	if df_name == "df":
		split_val = county.strip().split()

		if split_val[-1] == 'City':
			return county.upper() 
		elif state == 'LA':
			return (county + " Parish").upper()
		elif state == 'AK':
			if county == "Skagway":
				return (county + " Municipality").upper()
			elif county == "Wrangell":
				return (county + " City and Borough").upper()
			elif county in ["Aleutians West", "Southeast Fairbanks" , "Prince of Wales-Outer Ketchikan", "Bethel", "Hoonah Angoon", "Wade Hampton", "Kusilvak", "Nome", "Valdez-Cordova", "Wrangell-Petersburg", "Yukon-Koyukuk", "Dillingham", "Prince of Wales-Hyder"]:
				return (county + " Census Area").upper()
			else:
				return (county + " Borough").upper()
		else:
			return (county + " County").upper()

	elif df_name == "hazards":
		split_val = county.split("(")
		county_name = split_val[0].strip()
		if len(split_val) > 1:
			county_val = split_val[1][:-1].strip()
			return (county_name + " " + county_val).upper()
		else:
			return county_name.upper() 

	else:
		pass 

def update_df(df):

	df['state'] = df['state'].apply(lambda x: x.title())
	df['state_abbr'] = df['state'].apply(lambda x: state_abbr(x))
	df.drop(columns={'error', 'state'}, inplace=True)
	df.rename(columns={'state_abbr': 'state', 'year': 'Year'}, inplace=True)
	df['Year'] = df['Year'].astype(int)
	df['Year'] = df['Year'].apply(lambda x: update_year(x))
	df['county'] =  df.apply(lambda x: extract_county(x['county'], x['state'], "df"), axis=1)

	return df 


df = update_df(df)

################

# reading in hazards data 
hazards_df = pd.read_csv('data/HazardLocation_Year.csv')

natural_disasters = ["Severe Ice Storm", "Mud/Landslide", "Fire", "Severe Storm(s)", "Flood", "Hurricane", "Typhoon", "Earthquake", "Coastal Storm", "Tornado", "Snow", "Volcano"]
hazards_df = hazards_df.loc[(hazards_df['incidentType'].isin(natural_disasters)) & (hazards_df['incidentEndDate'].notna()), :]

# creating a df that counts number of disasters alerts in a year for a county 
hazards_df['length_of_decl'] = ((pd.to_datetime(hazards_df['incidentEndDate']) - pd.to_datetime(hazards_df['incidentBeginDate'])).astype("timedelta64[h]"))/24
number_of_dis = hazards_df.groupby(by=['state', 'Year','designatedArea'], as_index=False).agg({'disasterNumber': 'nunique', 'length_of_decl': 'mean'})
number_of_dis['county'] = number_of_dis.apply(lambda x: extract_county(x['designatedArea'], x['state'], "hazards"), axis=1)
number_of_dis.drop(columns={'designatedArea'}, inplace=True)

# filtering to only 2018 and 2019 data 
hazards_df = hazards_df.loc[hazards_df['Year'].isin([2014,2015,2016,2017,2018]), ['state','placeCode','designatedArea', 'Year']]
hazards_df['county'] =  hazards_df.apply(lambda x: extract_county(x['designatedArea'], x['state'], "hazards"), axis=1)
hazards_df.drop(columns={'designatedArea'}, inplace=True)

#hazards_df.to_csv('hazards_df.csv')
###############

# reading the population data 
pop_county = pd.read_csv('data/county_population.csv', encoding='latin-1')


def update_pop_df(df):
	df.rename(columns={'State':'state', 'County':'county'}, inplace=True)
	df.drop(columns={'ESTIMATESBASE2010'}, inplace=True)
	df = df.melt(id_vars=['state','county'], var_name='Year', value_name='population')
	df['Year'] = df['Year'].apply(lambda x: x[11:]).astype(int)
	df['state'] = df['state'].apply(lambda x: state_abbr(x))
	df['county'] = df['county'].apply(lambda x: x.upper())

	return df 

pop_county = update_pop_df(pop_county)

#pop_county.to_csv('pop.csv')
###############

# Connecting the four dataframes together. 

df_base = df.merge(hazards_df, on=['county','state','Year'], how='left')
df_base = df_base.merge(number_of_dis, on=['county','state','Year'], how='left')
df_base = df_base.merge(pop_county, on=['county','state','Year'], how='left')
df_base.drop_duplicates(inplace=True)
df_base.sort_values(by=['state','county','Year'], inplace=True)
df_base['hazards'] = np.where(df_base['placeCode'].notna(), 1,0)
df_base.drop(columns={'placeCode'}, inplace=True)
df_base['fmd'] = df_base['fmd'].apply(lambda x: x.split("%",1)[0].strip()).astype(float)
df_base['no_of_people'] = df_base.apply(lambda x: (x['fmd']/100)*x['population'], axis=1)
df_base_18 = df_base.loc[df_base['Year']==2018,:]
df_base = df_base.loc[df_base['Year']!=2018, :]

# calculating percent change betwen t-1, t+1, and t+2 

df_base['t+1_val'] = df_base.groupby(by=['state', 'county', 'FIPS'], as_index=False)['no_of_people'].shift(-1)
df_base['t+2_val'] = df_base.groupby(by=['state', 'county', 'FIPS'], as_index=False)['no_of_people'].shift(-2)

df_base['t_1'] = df_base.groupby(by=['state','county','FIPS'], as_index=False)['no_of_people'].pct_change()

df_base['t_1'] = df_base['t_1']/np.square(df_base['length_of_decl'])
df_base['t+1'] = df_base[['no_of_people','t+1_val']].diff(axis=1)['t+1_val']/(df_base['no_of_people'])
df_base['t+1'] = df_base['t+1']/df_base['length_of_decl']
df_base['t+2'] = df_base[['no_of_people','t+2_val']].diff(axis=1)['t+2_val']/(df_base['no_of_people'])
df_base['t+2'] = df_base['t+2']/np.sqrt(df_base['length_of_decl'])
df_base['avg_impact'] = df_base[['t_1','t+1','t+2']].sum(axis=1)/((1/df_base['length_of_decl'])+(1/np.sqrt(df_base['length_of_decl']))+(1/np.square(df_base['length_of_decl'])))
df_base['id'] = df_base['state']+"-"+df_base['county']


# Separating the data into two groups to find avg impact and then conduct KNN to fill in empty values 
hazard_df_base = df_base.loc[df_base['avg_impact'].notna(), :]
non_hazards_df = df_base.loc[~df_base['id'].isin(hazard_df_base['id']), ['state','county','id']].drop_duplicates()
non_hazards_df['avg_impact'] = np.nan 
## Grouping the data at the state and county level, and taking the weighted average of the avg_impact based on number of disasters in that year
wm = lambda x: np.average(x, weights=df_base.loc[x.index, 'disasterNumber'])
hazard_df_base = hazard_df_base.groupby(['state','county', 'id'], as_index=False).agg({'avg_impact': wm})


## Combining the two dataframes 
final_df = pd.concat([hazard_df_base, non_hazards_df])
final_df.sort_values(by=['state','county'], inplace=True)
final_df.reset_index(inplace=True)

## Combining this data with latest population data and fips data 
df_base_18.drop(columns={'Year','FIPS','disasterNumber','length_of_decl','hazards'}, inplace=True)
final_df.drop(columns={'index', 'id'}, inplace=True)
final_df = final_df.merge(df_base_18, on=['state','county'], how='left')
final_df.set_index(['state','county'], inplace=True)
index_values = final_df.index.values 


from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

imputer = KNNImputer(n_neighbors = 5, weights='distance')
impute_df = imputer.fit_transform(final_df)

final_df = pd.DataFrame(impute_df, columns=['avg_impact', 'fmd', 'population', 'no_of_people'], index=index_values)
final_df.reset_index(inplace=True)
final_df['state'] = final_df['index'].apply(lambda x: x[0].strip())
final_df['county'] = final_df['index'].apply(lambda x: x[1].strip())
final_df.drop(columns={'index'}, inplace=True)
final_df['avg_impact'] = final_df['avg_impact']*100 
final_df['mental_health_score'] = final_df['avg_impact']*final_df['fmd']

# Scaling mental health score to 1 to 100 
scaler = MinMaxScaler(feature_range=(1,100))
final_df['mental_health_score_scaled'] = scaler.fit_transform(final_df['mental_health_score'].to_numpy().reshape(-1,1))
final_df = final_df.drop_duplicates()
final_df.to_csv('mental_health_score.csv')
