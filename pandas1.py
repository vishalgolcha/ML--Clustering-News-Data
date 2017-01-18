import quandl
import pandas as pd 

api_key= 'bjy2zSTV6VgrdnoNFNdk'
# df= quandl.get('FMAC/HPI_AK',authtoken=api_key)

print(df.head())
fiddy_states=pd.read_html('https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States')
# print(fiddy_states[0][1])

for abbv in fiddy_states[0][1][1:]:
	query="FMAC/HPI_"+str(abbv)
	df=quandl.get(query,authtoken=api_key)

	if main_df.empty:
		main_df=df
	else:
		main_df=main_df.join(df)

print(main_df.head())