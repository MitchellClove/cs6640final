import pandas as pd
from sklearn.model_selection import train_test_split


def clean(name):
    train_data = pd.read_csv(f"../dataset/regression/melting/{name}.csv")

    smiles_train = train_data["SMILES"]
    tms_train = train_data["Tm" if name in ["train","Bradley_Melting_Point_Dataset"] else "Group 1"]

    smile_tms_train = zip(smiles_train,tms_train)
    with open(f"../dataset/regression/melting/data_{name}.txt", 'w') as f:
        f.write("SMILES,Tm\n")
        for tup in smile_tms_train:
            f.write(f"{tup[0]} {tup[1]}\n")

clean("train")
clean("test")
clean("Bradley_Melting_Point_Dataset")

train_csv = pd.read_csv("../dataset/regression/melting/train.csv")
test_csv = pd.read_csv("../dataset/regression/melting/test.csv")
X = train_csv["SMILES"]
y = train_csv['Tm']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)
zipped_valid = zip(X_test,y_test)
with open("valid.txt",'w') as f:
    for x in zipped_valid:
        f.write(f"{x[0]} {x[1]}\n")


