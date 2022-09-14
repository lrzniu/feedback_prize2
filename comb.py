import pandas as pd
df = pd.read_csv("pseudolabel_output.csv")
labels = ['Adequate', 'Effective', 'Ineffective']
df['discourse_effectiveness']=df[labels].idxmax(axis=1)
del df['Adequate'],df['Effective'],df['Ineffective']
p=df['essay_text']
del df['essay_text']
df['eassy_text']=p
df.to_csv('pseudolabel_output_same2.csv', index=False)
