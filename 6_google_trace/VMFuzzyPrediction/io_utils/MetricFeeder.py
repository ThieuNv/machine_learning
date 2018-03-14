from __init__ import *
from sklearn.preprocessing import MinMaxScaler
from utils.SlidingWindowUtil import SlidingWindow
class MetricFeeder:
    def __init__(self,skip_lists=1,split_size=None):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.skip_lists = skip_lists
        if(split_size!=None):
            self.split_size = split_size
        # self.data = pd.read_hdf(self.metric_type.get(metric_type))["Volume"]
        self.result = {}
        self.metric_type = {
            "cpu_util": "../data/vdc/sample_cpu_util.json",
            "disk_write_rate": "../data/vdc/sample_disk_write.json",
            "disk_read_rate": "../data/vdc/sample_disk_read.json",
            "network_incoming_rate": "../data/vdc/sample_network_incoming.json",
            "network_outgoing_rate": "../data/vdc/sample_network_outgoing.json"
        }

    def fetch(self, metrics, n_sliding_window, range_fetch=None):
        data_fetch_X = []
        data_fetch_y = []
        for metric in metrics:
            data = self.average_metric(pd.read_json(self.metric_type[metric])["Volume"],skip_lists=self.skip_lists)
            self.result[metric] = data
            data_fetch_X.append(self.fetch_metric_train(data, n_sliding_window,range_fetch))
            data_fetch_y.append(self.fetch_metric_test(data, n_sliding_window,range_fetch))
        X_test = np.asarray([np.array(t, dtype=np.float32).flatten().tolist() for t in zip(*data_fetch_X)])
        y_test = np.asarray([np.array(t).flatten().tolist() for t in zip(*data_fetch_y)])
        return X_test, y_test
    def _fetch(self, n_sliding_window, range_fetch=None):
        data_fetch_X = []
        data_fetch_y = []
        for metric,data in self.result.iteritems():
            # data = self.average_metric(data,skip_lists=self.skip_lists)
            data_fetch_X.append(self.fetch_metric_train(data, n_sliding_window,range_fetch))
            data_fetch_y.append(self.fetch_metric_test(data, n_sliding_window,range_fetch))
        X_test = np.asarray([np.array(t, dtype=np.float32).flatten().tolist() for t in zip(*data_fetch_X)])
        y_test = np.asarray([np.array(t).flatten().tolist() for t in zip(*data_fetch_y)])
        return X_test, y_test

    def split_train_and_test(self,metrics=None,n_sliding_window=4,train_size = 0.7):
        length_data = 0
        if metrics == None:
            metrics = self.metric_type.keys()
            for metric in metrics:
                self.result[metric]= self.average_metric(pd.read_json(self.metric_type[metric])["Volume"],skip_lists=self.skip_lists)
                length_data = self.result[metric].shape[0]
        point = int(length_data*train_size)
        range_train = (-1,point)
        range_test = (point,-1)
        X_train, y_train = self._fetch(n_sliding_window,range_train)
        X_test, y_test = self._fetch(n_sliding_window,range_test)
        return X_train, y_train,  X_test, y_test

    def fetch_metric_train(self, data, n_sliding_window, range_fetch):
        from_range = range_fetch[0]
        to_range = range_fetch[1]
        range_data = None
        if(from_range==-1):
            range_data = data[:to_range].reshape(-1, 1)
        else:
            range_data = data[from_range:to_range].reshape(-1, 1)
        result = list(SlidingWindow(
            self.scaler.fit_transform(range_data),
            n_sliding_window))
        return result

    def fetch_metric_test(self, data, n_sliding_window, range_fetch):
        from_range = range_fetch[0]
        to_range = range_fetch[1]
        if(to_range==-1):
            range_data = data[from_range + n_sliding_window:].reshape(-1, 1)
        else:
            range_data = data[from_range + n_sliding_window:to_range].reshape(-1, 1)
        result = list(
            self.scaler.fit_transform(range_data))
        return result

    def convert(self, data_scale, type="cpu_util"):
        min = self.result[type].min()
        max = self.result[type].max()
        return data_scale * (max - min) + min
    def average_metric(self,data,skip_lists=None):
        # skip_lists = 5
        # idx = 0
        if(skip_lists==None):
            skip_lists = self.skip_lists
        # avg_cpu = pd.Series(np.zeros(data.shape[0]/skip_lists))
        return pd.Series([data[idx] for idx in np.arange(0,data.shape[0],step=skip_lists)])
