from sklearn.preprocessing import LabelEncoder
# encoding categorical columns 

def preprocessed(df):
 encode = LabelEncoder()

 df['Extracurricular Activities'] = encode.fit_transform(df['Extracurricular Activities'])
 
 for col in df.columns:
     df[col] = df[col].astype('int32')

 return df