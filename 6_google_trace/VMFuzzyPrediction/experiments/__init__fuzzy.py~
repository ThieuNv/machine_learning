import math
from utils.SlidingWindowUtil import SlidingWindow
from sklearn.preprocessing import MinMaxScaler
from estimators.GAEstimator import GAEstimator
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from estimators.NeuralFlow import *
from utils.GraphUtil import *
from sklearn import datasets, metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.GraphUtil import *
from estimators.GAEstimator import GAEstimator
from io_utils.NumLoad import *
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from scaling.ProactiveSLA import ProactiveSLA
from io_utils.GFeeder import GFeeder

print "Loading Data"

metric = {
	"mem_usage":0.0308,
	"cpu_rate":0.25
}
metric_type = "cpu_rate"
scaler = MinMaxScaler()
dat = pd.read_csv('sampling_617685_metric_10min_datetime.csv', parse_dates=True, index_col=0)[:3000]
dat = pd.Series(dat[metric_type].round(3))
distance = round(dat.max() / (dat.max() / metric[metric_type] + 2),4)
print "Fuzzy distance = %s"%distance

partition_size = distance
umin = math.floor(min(dat));
umax = math.ceil(max(dat));
# 2: Partition of universe
# Method: Dividing in the half-thousands
def get_midpoint(ptuple):
    return 0.5*(ptuple[0]+ptuple[1])
def get_midpoint_vector(tuple_vector):
    return [get_midpoint(x) for x in tuple_vector];
def get_fuzzy_class(point, partition_size):
    return int(math.floor(point / partition_size))
def get_fuzzy_dataset(data):
    u_class = []
    for item in data:
        u_class.append(get_fuzzy_class(item,partition_size))
    return u_class
def mapping_class(u_class):
    unique_class = np.unique(u_class)
    index = np.arange(unique_class.shape[0])
    inverted = {}
    mapping = {}
    for idx,val in enumerate(unique_class):
        mapping[val] = idx
        inverted[idx] = val
    return mapping, inverted
def defuzzy(index, inverted,midpoints):
    f_class = inverted[index % len(inverted)]
    return midpoints[f_class]

# get fuzzy class set
nIter = int((umax-umin)/partition_size)
u_vectorized = []

for i in range(nIter) :
    u_vectorized.append((umin + i*partition_size,umin + (i+1)*partition_size));

u_midpoints = get_midpoint_vector(u_vectorized)
u_class = np.array(get_fuzzy_dataset(dat),dtype=np.int32)

u_unique_inverted, u_unique_mapping = mapping_class(u_class)
u_class_transform = [u_unique_inverted[item] for item in u_class]

sliding_number = 3
# result = []
X_train_size = int(len(u_class_transform)*0.7)
sliding = np.array(list(SlidingWindow(u_class_transform, sliding_number)))
sliding = np.array(sliding, dtype=np.int32)
X_train = sliding[:X_train_size]
y_train = u_class_transform[sliding_number:X_train_size+sliding_number]
X_test = sliding[X_train_size:]
y_test = u_class_transform[X_train_size+sliding_number-1:]
y_actual_test = dat[X_train_size+sliding_number-1:].tolist()

np.savez("fuzzy_train_direct_RAM",X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
