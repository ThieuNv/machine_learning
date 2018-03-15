from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
class MinMaxScalerIO:
    def __init__(self,range=[0,1]):
        self.scaler = []
    def fit_transform(self,data_transform,attributes=None):
        self.scaler = [MinMaxScaler() for i in len(attributes)]
        self.attr = attributes
        for idx,attribute in enumerate(attributes):
            data_transform[attribute] = self.scaler[idx].fit_transform(data_transform[attribute])
        return data_transform
    # def inverse_transform(self):
    #     for idx, attribute in enumerate(attributes):
    #         data_transform[attribute] = self.scaler[idx].fit_transform(data_transform[attribute])