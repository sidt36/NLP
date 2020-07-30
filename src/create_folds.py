import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold

if __name__ =='__main__':



    df = pd.read_csv(r"input\IMDB Dataset.csv")
    df.sentiment = df.sentiment.apply(
    lambda x: 1 if x == "positive" else 0
    )

    skf = StratifiedKFold(n_splits=5)


    y = df.sentiment
    df = df.sample(frac=1).reset_index(drop=True)


    for fold , (tr,val) in enumerate(skf.split(X=df, y=y)):
        df.loc[val,'kfold'] = fold


    df.to_csv(r'C:\Users\Sidharth Tadeparti\Documents\DataScienceProjects\Project\input\imdb_folds.csv',index=False)


