#!/usr/bin/env python3
import math
from anytree import AnyNode, RenderTree, LevelOrderIter
from anytree.render import AsciiStyle
import pandas as pd
import numpy as np
from pprint import pprint
from chefboost.training import Training

final = []
cols = ["股價報酬率", "負債佔比", "長期資金佔不動產.廠房及設備比率", "現金占比", "應收帳款占比", "存貨占比", "權益占比", "現金流量比", "流動比率", "速動比率", "利息保障倍數", "應收款項週轉率", "平均收現日數", "存貨週轉率", "應付款項週轉率", "平均銷貨日數", "平均付現日數", "不動產.廠房及設備週轉率", "總資產週轉率", "資產報酬率", "權益報酬率", "純益率", "每股盈餘"]

def metric(samples, feat, target="股價報酬率"):
    samples.sort_values(by=[feat], inplace=True)
    samples["avg"] = samples[feat].rolling(window=2).mean()

    records = []

    for threshold in [t for t in samples["avg"] if math.isnan(t) == False]:
        df1 = samples.where(samples[feat] >= threshold)
        mean1 = samples.where(samples[feat] >= threshold)[target].mean()
        df2 = samples.where(samples[feat] < threshold)
        mean2 = samples.where(samples[feat] < threshold)[target].mean()
        shape1 = df1[target].dropna().shape
        err1 = np.sum((df1[target].dropna() - np.full(shape1, mean1))**2)
        shape2 = df2[target].dropna().shape
        err2 = np.sum((df2[target].dropna() - np.full(shape2, mean2))**2)
        records.append((threshold, err1 + err2))
    return sorted(records, key=lambda x: x[1])[0]


def id3(subset, origin, features, node, target="股價報酬率"):
    if len(subset) <= 3:
        node.isleaf=True
        return
    if len(features) == 0: return node

    rank = sorted([(feat, metric(subset, feat)) for feat in features], key=lambda x:x[1][1])
    selected_feat = rank[0][0]
    threshold = rank[0][1][0]
    df1 = subset.where(subset[selected_feat] >= threshold).dropna(subset=[target])
    df2 = subset.where(subset[selected_feat] < threshold).dropna(subset=[target])
    features.remove(selected_feat)

    node.description = F"{selected_feat} < {threshold:.2f}"
    node.criterion = selected_feat
    node.threshold = threshold

    node_a = AnyNode(parent=node, description="", st=True, criterion=None, threshold=None, average=df1[target].mean(), isleaf=False)
    node_b = AnyNode(parent=node, description="", st=False, criterion=None, threshold=None, average=df2[target].mean(), isleaf=False)

    id3(df2, origin, features, node_a)
    id3(df1, origin, features, node_b)

    return node


def predict(current, data, stop=True):
    for idx, row in data.iterrows():
        node = current
        while not node.isleaf:
            print(node.description)
            if row[cols.index(node.criterion)] < node.threshold:
                node = node.children[0]
            else:
                node = node.children[1]
        print(F"Predicted value: {node.average:.3f}")


if "__main__" == __name__:
    dataset = pd.read_csv("dataset.csv", names=cols, skiprows=1)

    root = AnyNode(id="root", description="", st=None, criterion=None, threshold=None, average=None, isleaf=False)
    tree = id3(dataset, dataset, cols[1:], root)
    print(RenderTree(tree, style=AsciiStyle()))

    # Predict
    predict(tree, dataset.head())
