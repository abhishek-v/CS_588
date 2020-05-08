#Code to modify dataset output classes based on probability
import pandas as pd
from random import choices


if(__name__ == "__main__"):

    df = pd.read_csv("./adult.csv")
    sal = list()

    print(df["education"].unique())
    grad = ["Assoc-acdm","Doctorate","Masters","Prof-school"]
    bac = ["Assoc-voc","Bachelors"]
    count = 0
    for i in range(len(df)):
        #print(i)
        if(df.iloc[i,5] in bac):
            a = [0,1]
            p = [0.1,0.95]
            if(choices(a,p)==[1]):
                #df.at[i,'salary_class'] = "<=80K"
                sal.append("<=80K")
            else:
                # df.at[i,'salary_class'] = "<50K"
                sal.append("<50K")
        elif(df.iloc[i,5] in grad):
            a = [0,1]
            p = [0.01,0.99]
            if(choices(a,p)==[1]):
                # df.at[i,'salary_class'] = ">100K"
                sal.append(">100K")
            else:
                # df.at[i,'salary_class'] = "<=80K"
                sal.append("<=80K")
        else:
            a = [0,1]
            p = [0.001,0.999]
            if(choices(a,p)==[1]):
                # df.at[i,'salary_class'] = "<50K"
                sal.append("<50K")
            else:
                count = count + 1
                print(count)
                # df.at[i,'salary_class'] = ">100K"
                sal.append(">100K")
    print(len(sal))
    df = df.drop(columns=['salary_class'])
    df.insert(9,'salary_class',sal)
    df.to_csv("adult2.csv",index=False)
    print("Dataset generated successfully")
