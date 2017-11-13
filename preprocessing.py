import json
import pandas as pd
import sys
if __name__=="__main__":
    #Taking input file as argument
    inputfile=sys.argv[1]
    with open(inputfile) as input_file:
        json_data=[json.loads(line) for line in input_file]
    #Fetching only photo id and label id and converting it into csv file
    result=[]
    for item in json_data:
        my_dict={}
        my_dict['photoid']=item.get('photo_id')+'.jpg'
        my_dict['label']=item.get('label')
        result.append(my_dict)
    df=pd.DataFrame(result)
    df=df.reindex(columns=['photoid','label'])
    #Encoding categorical data
    from sklearn.preprocessing import LabelEncoder
    labelencoder=LabelEncoder()
    df.ix[:,1]=labelencoder.fit_transform(df.ix[:,1])
    df.to_csv('out.csv',index=False)