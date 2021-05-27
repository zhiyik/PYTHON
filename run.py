#!/usr/bin/env python3
# pip install anytree pandas
import math
from anytree import AnyNode, RenderTree, LevelOrderIter
from anytree.exporter import UniqueDotExporter
from anytree.dotexport import RenderTreeGraph
from anytree.render import AsciiStyle
import pandas as pd
import numpy as np
from pprint import pprint

final = []
cols = ["股價報酬率", "負債佔比", "長期資金佔不動產.廠房及設備比率", "現金占比", "應收帳款占比", "存貨占比", "權益占比", "現金流量比", "流動比率", "速動比率", "利息保障倍數", "應收款項週轉率", "平均收現日數", "存貨週轉率", "應付款項週轉率", "平均銷貨日數", "平均付現日數", "不動產.廠房及設備週轉率", "總資產週轉率", "資產報酬率", "權益報酬率", "純益率", "每股盈餘"]

def metric(samples, feat, target="股價報酬率"):
    """ 對每一個 subset 的 每一個 feature 計算 股價報酬率的 MSE
    原來的資料是有限類別的，因此可以針對每種類別的可能性計算機率與 entropy
    但這裡的 target 報酬率是 continous data, 因此為了 fit data, 選擇較常使用的 Mean Square Error
    """
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
    """ID3 algorithm

    初始的 subset 是原來的 training set, 會根據剛剛算出來的 metric 切分資料，別分別在傳入 id3()
    這個迭代會直到一些條件達成，這裡是設計 當有 subset 少於三筆資料就停止，並判斷結果是這些資料的 投資報酬率的平均值
    """
    if len(subset) <= 3:
        node.isleaf=True
        node.name = F"{node.average:.2f}"
        return
    if len(features) == 0: return

    rank = sorted([(feat, metric(subset, feat)) for feat in features], key=lambda x:x[1][1])
    selected_feat = rank[0][0]
    threshold = rank[0][1][0]
    df1 = subset.where(subset[selected_feat] >= threshold).dropna(subset=[target])
    df2 = subset.where(subset[selected_feat] < threshold).dropna(subset=[target])
    features.remove(selected_feat)

    node.name = F"{selected_feat} < {threshold:.2f}"
    node.criterion = selected_feat
    node.threshold = threshold

    node_a = AnyNode(parent=node, name="", st=True, criterion=None, threshold=None, average=df1[target].mean(), isleaf=False)
    node_b = AnyNode(parent=node, name="", st=False, criterion=None, threshold=None, average=df2[target].mean(), isleaf=False)

    id3(df2, origin, features, node_a)
    id3(df1, origin, features, node_b)

    return node


def predict(current, data, stop=True):
    for idx, row in data.iterrows():
        node = current
        while not node.isleaf:
            print(node.name)
            if row[cols.index(node.criterion)] < node.threshold:
                node = node.children[0]
            else:
                node = node.children[1]
        print(F"Predicted value: {node.average:.3f}")


if "__main__" == __name__:
    dataset = pd.read_csv("new_dataset.csv", names=cols, skiprows=1)

    # Init root node
    root = AnyNode(id="root", name="", st=None, criterion=None, threshold=None, average=None, isleaf=False)
    tree = id3(dataset, dataset, cols[1:], root)
    print(RenderTree(tree, style=AsciiStyle()))

    # Predict
    predict(tree, dataset.head())

    RenderTreeGraph(root).to_picture("tree.png")

    # $ python run.py
    # AnyNode(average=None, criterion='速動比率', name='速動比率 < 163.92', id='root', isleaf=False, st=None, threshold=163.92219850000004)
    # |-- AnyNode(average=0.84351145, criterion='存貨占比', name='存貨占比 < 26.80', isleaf=False, st=True, threshold=26.8)
    # |   |-- AnyNode(average=-0.08721644585714285, criterion='現金流量比', name='現金流量比 < 37.80', isleaf=False, st=True, threshold=37.8)
    # |   |   |-- AnyNode(average=-0.09049773800000001, criterion='資產報酬率', name='資產報酬率 < 1.80', isleaf=False, st=True, threshold=1.7980808364999998)
    # |   |   |   |-- AnyNode(average=0.12472465025, criterion=None, name='', isleaf=True, st=True, threshold=None)
    # |   |   |   +-- AnyNode(average=0.24460932000000002, criterion='應收帳款占比', name='應收帳款占比 < 23.75', isleaf=False, st=False, threshold=23.75)
    # |   |   |       |-- AnyNode(average=0.04076087, criterion=None, name='', isleaf=True, st=True, threshold=None)
    # |   |   |       +-- AnyNode(average=0.152712577, criterion=None, name='', isleaf=True, st=False, threshold=None)
    # |   |   +-- AnyNode(average=0.16468620683333332, criterion=None, name='', isleaf=True, st=False, threshold=None)
    # |   +-- AnyNode(average=0.12823135757142856, criterion='平均銷貨日數', name='平均銷貨日數 < 75.14', isleaf=False, st=False, threshold=75.14115065000001)
    # |       |-- AnyNode(average=0.021336732666666667, criterion='利息保障倍數', name='利息保障倍數 < 114068.82', isleaf=False, st=True, threshold=114068.82097500001)
    # |       |   |-- AnyNode(average=0.054716981, criterion=None, name='', isleaf=True, st=True, threshold=None)
    # |       |   +-- AnyNode(average=-0.24308076666666667, criterion=None, name='', isleaf=True, st=False, threshold=None)
    # |       +-- AnyNode(average=-0.16863132975, criterion=None, name='', isleaf=True, st=False, threshold=None)
    # +-- AnyNode(average=0.02050745585714286, criterion=None, name='', isleaf=True, st=False, threshold=None)
    # 速動比率 < 163.92
    # 存貨占比 < 26.80
    # 現金流量比 < 37.80
    # 資產報酬率 < 1.80
    # Predicted value: 0.125
    # 速動比率 < 163.92
    # 存貨占比 < 26.80
    # 現金流量比 < 37.80
    # 資產報酬率 < 1.80
    # 應收帳款占比 < 23.75
    # Predicted value: 0.041
    # 速動比率 < 163.92
    # 存貨占比 < 26.80
    # 平均銷貨日數 < 75.14
    # 利息保障倍數 < 114068.82
    # Predicted value: 0.055
    # 速動比率 < 163.92
    # 存貨占比 < 26.80
    # 現金流量比 < 37.80
    # 資產報酬率 < 1.80
    # 應收帳款占比 < 23.75
    # Predicted value: 0.041
    # 速動比率 < 163.92
    # 存貨占比 < 26.80
    # 現金流量比 < 37.80
    # Predicted value: 0.165
