import math
from os import major
import re
from sqlite3 import paramstyle
from typing import List, Dict, Any, Optional, Tuple

from numpy import partition

class TreeNode:
    def __init__(self) -> None:
        self.label: Optional[str] = None #储存结论（如果能够得到）
        self.split_feature: Optional[str] = None #储存划分属性
        self.children: Optional[Dict[Any, 'TreeNode']] = None #储存子树
        self.majority_label: Optional[str] = None #储存多数类标签
    
    def is_leaf(self) -> bool:#判断是否为叶子节点
        return self.label is not None


    
class ID3Tree:
    def __init__(self) -> None:
        self.root: Optional[TreeNode] = None

    def train(self, data: List[Dict[str,str]], labels: List[str]) -> "ID3Tree":
        feature_name: List[str] = list(data[0].keys())
        self.root = self._build_tree(data, labels, feature_name)
        return self
    
    def predict(self, data: List[Dict[str,str]]) -> List[str]:
        return [self._predict_single(instance) for instance in data]
    
    def _predict_single(self, instance:Dict[str, str]) -> str:
        node = self.root
        while not node.is_leaf():
            feature_value = instance.get(node.split_feature)
            if feature_value not in node.children:
                return self.mojority_label
            node = node.children[feature_value]
        return node.label



    def _build_tree(self, data: List[Dict[str,str]], labels: List[str], remaining_features: List[str]) -> TreeNode:
        node = TreeNode()
        self.mojority_label= self._get_majority(labels)
        
        if (self._all_same(labels)):
            node.label = labels[0]
            return node
        
        if len(remaining_features) == 0:
            node.label = self.mojority_label
            return node
        
        best_feature = self._best_feature(data, labels, remaining_features)
        node.split_feature = best_feature
        node.children = {}

        partitions = self._partition_by_feature(data, labels, best_feature)
        next_features = [f for f in remaining_features if f != best_feature]

        for feature_value, (sub_data, sub_labels) in partitions.items():
            child_node = self._build_tree(sub_data, sub_labels, next_features)
            node.children[feature_value] = child_node
        
        return node
        
    def _all_same(self, labels: List[str]) -> bool:
        return all(label == labels[0] for label in labels)
    
    def _get_majority(self, labels: List[str]) -> str:
        counts={}
        majority_label = None

        for lable in labels:
            counts[lable] = counts.get(lable, 0) + 1
        
        majority_label = max(counts, key=counts.get)
        return majority_label


    def _entropy(self, lables: List[str])-> float:
        counts = {}
        total = len(lables)

        for label in lables:
            counts[label] = counts.get(label, 0) + 1

        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)

        return entropy
    
    def _partition_by_feature(self, data: List[Dict[str,str]], labels: List[str], feature: str) -> Dict[str, Tuple]:
        partitions = {}
        for instance, label in zip(data, labels):
            feature_value = instance[feature]
            if feature_value not in partitions:
                partitions[feature_value] = ([], [])
            partitions[feature_value][0].append(instance)
            partitions[feature_value][1].append(label)
        return partitions
    
    def _best_feature(self, data: List[Dict[str,str]], labels: List[str], features: List[str]) -> str:
        best_f = features[0]
        best_gain = -1.0
        
        for feature in features:
            gain = self._information_gain(data, labels, feature)
            if gain > best_gain:
                best_gain = gain
                best_f = feature
        return best_f
    
    def _information_gain(self, data: List[Dict[str,str]], labels: List[str], feature: str) -> float:
        original_entropy = self._entropy(labels)
        partitions=self._partition_by_feature(data, labels, feature)
        total = len(labels)
        parted_entropy = 0.0
        
        for syb_data, sub_lable in partitions.values():
            weight = len(sub_lable) / total
            parted_entropy += weight * self._entropy(sub_lable)
        return original_entropy - parted_entropy
