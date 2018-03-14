import pandas as pd
from estimators.FuzzyFlow import FuzzyFlow
fuzzy = FuzzyFlow()
dat = pd.read_csv('../sampling_617685_metric_10min_datetime.csv',parse_dates=True,index_col=0)[:3000]
dat = pd.Series(dat['cpu_rate'].round(3))
fuzzy.fit_transform(dat)