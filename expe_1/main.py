import pandas as pd
from typing import List, Dict, Any
from id3_tree import ID3Tree

if __name__ == "__main__":
    train_file = pd.read_csv("train.csv")
    predict_file = pd.read_csv("predict.csv")

    #读取倒数最后一列之前
    train_data = train_file.iloc[:, :-1].astype(str).to_dict(orient="records")
    train_labels = train_file.iloc[:, -1].astype(str).tolist()

    tree = ID3Tree()
    tree.train(train_data, train_labels)

    predict_data = predict_file.iloc[:, :-1].astype(str).to_dict(orient="records")
    predictions = tree.predict(predict_data)
    
    output = predict_file.copy()
    #将预测结果加到最后一列（新加一列）
    output["label"] = predictions
    output.to_csv("output.csv", index=False)