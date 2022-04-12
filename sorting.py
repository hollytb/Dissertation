import pandas as pd

files = ['coefClassMagazine.csv', 'coefClassNational.csv', 'coefClassSports.csv']
topics = ['Magazine', 'National', 'Sports']
i = 0
for file in files:
    curr_df = pd.read_csv(file)
    rslt_df = curr_df.sort_values(by='coefficient', ascending=False)
    rslt_df.to_csv(path_or_buf=("sortedClass" + str(topics[i]) + ".csv"),  index=False,sep='&', float_format='%.6f')
    i += 1
