from utils.SlidingWindowUtil import SlidingWindow
from __init__ import *
import os
class GFeeder:
    def __init__(self,skip_lists=1,split_size=None,normalize_space=True, file_name="data/gdata/gcluster_normalize_space.json"):
        self.skip_lists = skip_lists
        if(split_size!=None):
            self.split_size = split_size
        self.result = {}
        self.CPU_UTIL = 'cpu_usage'
        self.DISK_IO_TIME = 'disk_io_time'
        self.DISK_SPACE = 'disk_space'
        self.MEM_USAGE = 'mem_usage'
        if(normalize_space==False):
            self.file_name = "data/gdata/gcluster_1268205_1min.json"
        else:
            self.file_name = file_name
        self.metric_type = [
            self.CPU_UTIL, self.MEM_USAGE, self.DISK_IO_TIME,self.DISK_SPACE
        ]
        self.metrics = self.metric_type
        current_folder_path, current_folder_name = os.path.split(os.getcwd())
        if(current_folder_name!='VMResourcePrediction'):
            self.file_name = '../%s'%file_name
    def read(self,metrics=None):
        if (metrics!=None):
            self.metrics = metrics

        return self._skip_windows(pd.read_json(self.file_name,orient='records',dtype=float)[self.metrics][:1152614])
    def fetch_metric_train(self,data,n_sliding_window,range_fetch):
        from_range = range_fetch[0]
        to_range = range_fetch[1]
        # range_data=None
        if(from_range==-1):
            range_data = data[:to_range]
        else:
            range_data = data[from_range:to_range]
        result = list(SlidingWindow(range_data,n_sliding_window))
        return result
    def fetch_metric_test(self, data, n_sliding_window, range_fetch):
        from_range = range_fetch[0]
        to_range = range_fetch[1]
        range_data=None
        if(to_range==-1):
            range_data = data[from_range + n_sliding_window:].reshape(-1, 1)
        else:
            range_data = data[from_range + n_sliding_window:to_range].reshape(-1, 1)
        result = np.array(range_data)
        return result
    def _fetch(self,data,n_sliding_window,range_fetch=None):
        data_fetch_X = []
        data_fetch_y = []
        tmp_iter = data
        if(isinstance(data,pd.DataFrame) or isinstance(data,pd.Series)):
            tmp_iter = data.iteritems()
        for column,data in tmp_iter:
            data_fetch_X.append(self.fetch_metric_train(data, n_sliding_window,range_fetch))
            data_fetch_y.append(self.fetch_metric_test(data, n_sliding_window,range_fetch))
        X = np.asarray([np.array(t).flatten().tolist() for t in zip(*data_fetch_X)])
        y = np.asarray([np.array(t).flatten().tolist() for t in zip(*data_fetch_y)])
        return X, y
    def _fetch_metric_window(self,metrics_windows,range_fetch=None):
        data_fetch_X = []
        data_fetch_y = []
        for column,data in self.result.iteritems():
            data_fetch_X.append(self.fetch_metric_train(data, metrics_windows[column],range_fetch))
            data_fetch_y.append(self.fetch_metric_test(data, metrics_windows[column],range_fetch))
        X_test = np.asarray(self._concat(zip(*data_fetch_X)))
        y_test = np.asarray(self._concat(zip(*data_fetch_y)))
        return X_test, y_test
        # return data_fetch_X,data_fetch_y
    def split_train_and_test(self,data=None,metrics=None,n_sliding_window=4,train_size = 0.7):
        allowed_classes = (pd.Series, pd.DataFrame)
        if(data.empty):
            self.result = self.read(metrics)
        if isinstance(data, allowed_classes):
            self.result = data[metrics]
            self.CPU_UTIL = 'cpu_rate'
        length_data = self.result.shape[0]
        point = int(length_data*train_size)
        range_train = (-1,point)
        range_test = (point,-1)
        X_train, y_train = self._fetch(self.result,n_sliding_window,range_train)
        X_test, y_test = self._fetch(self.result,n_sliding_window,range_test)
        self.input_size = len(metrics)*n_sliding_window
        self.output_size = len(metrics)
        return X_train, y_train,  X_test, y_test
    def _skip_windows(self,data):
        new_indices = np.arange(data.shape[0],step=self.skip_lists)
        return pd.DataFrame(data.iloc[idx] for idx in new_indices)
    def split_train_and_test_window(self,metrics_windows=None,train_size = 0.7):
        self.result = self.read(metrics=metrics_windows.keys())
        length_data = self.result.shape[0]
        point = int(length_data*train_size)
        range_train = (-1,point)
        range_test = (point,-1)
        # return self._fetch_metric_window(metrics_windows,range_train)
        X_train, y_train = self._fetch_metric_window(metrics_windows,range_train)
        X_test, y_test = self._fetch_metric_window(metrics_windows,range_test)
        self.input_size = sum(metrics_windows.values())
        self.output_size = len(metrics_windows)
        return X_train, y_train,  X_test, y_test
    def _concat(self,zip_array):
        big_one = []
        for arr in zip_array:
            tmp = arr[0].tolist()
            for after in arr[1:]:
                for t in after:
                    tmp.append(t)
            big_one.append(tmp)
        return big_one