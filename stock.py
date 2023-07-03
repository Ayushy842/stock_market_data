from nsetools import Nse
import pandas as pd
nse = Nse()
print(nse)
stock_data = nse.get_quote('IOC.ns')
df = pd.DataFrame.from_dict(stock_data, orient='index').T
print(df)


  

