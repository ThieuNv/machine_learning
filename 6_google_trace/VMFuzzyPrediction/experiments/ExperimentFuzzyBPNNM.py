from io_utils.NumLoad import *
from sklearn.cross_validation import train_test_split
from estimators.FuzzyFlow import FuzzyFlow
from utils.TrainingTestMaker import TrainingTestMaker
from scaling.ProactiveSLA import ProactiveSLA
from __init__fuzzy import *

dataset_holder = []
trainee_holder = {}
metrics = ["cpu_rate","mem_usage","disk_io_time"]
arr_desk = ['X_train','y_train','X_test']
sliding_number = 3
data = pd.read_csv('sampling_617685_metric_10min_datetime_origin.csv',parse_dates=True,index_col=0)[:3000]
def experiment(sliding_number):
    for metric in metrics:
        dat = pd.Series(data[metric].round(5))
        fuzzy_engine = FuzzyFlow()
        data_maker = TrainingTestMaker()
        fuzzy_engine.fit_transform(dat)
        sliding = np.array(list(SlidingWindow(fuzzy_engine.u_class_transform, sliding_number)))
        X_train, y_train, X_test, y_test = data_maker.make_fuzzy_test(sliding, fuzzy_engine.u_class_transform, dat)
        dataset_holder.append(fuzzy_engine)
        trainee_holder[metric] = {
            'X_train': X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }
    y_train = np.asarray(zip(*[trainee_holder[metric]['y_train'] for metric in metrics]))
    # X_test = zip(trainee_holder['cpu_rate']['X_test'],trainee_holder['mem_usage']['X_test'])
    X_train = []
    X_test = []
    # y_train = []
    for i in np.arange(len(trainee_holder['cpu_rate']['X_train'])):
    #     tmp = zip(trainee_holder['cpu_rate']['X_train'][i],trainee_holder['mem_usage']['X_train'][i])
        tmp = zip(*[trainee_holder[metric]['X_train'][i] for metric in metrics])
        X_train.append(np.ravel(tmp))
    for i in np.arange(len(trainee_holder['cpu_rate']['X_test'])):
        tmp = zip(*[trainee_holder[metric]['X_test'][i] for metric in metrics])
        X_test.append(np.ravel(tmp))
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    classifier = KerasRegressor(hidden_nodes=[64],steps=10000,batch_size=32, activation='sigmoid',verbose=2, learning_rate=0.001)
    a = classifier.fit(X_train, y_train)
    y_pred = np.round(abs(classifier.predict(X_test)))
    y_cpu = dataset_holder[0].inverse_transform(abs(y_pred[:, 0]))
    y_ram = dataset_holder[1].inverse_transform(abs(y_pred[:, 1]))
    y_disk_io = dataset_holder[1].inverse_transform(abs(y_pred[:, 2]))
    score_mae_CPU = mean_absolute_error(y_cpu, trainee_holder['cpu_rate']['y_test'])
    score_mae_RAM = mean_absolute_error(y_ram, trainee_holder['mem_usage']['y_test'])
    score_mae_disk_io = mean_absolute_error(y_ram, trainee_holder['disk_io_time']['y_test'])
    y_test = zip(trainee_holder['cpu_rate']['y_test'],trainee_holder['mem_usage']['y_test'])
    np.savez('model_saved/Fuzzy_BPNNM_%s_%s' % (sliding_number, score_mae_CPU), y_pred=y_pred, y_true=y_test)
    return sliding_number, score_mae_CPU, score_mae_RAM, score_mae_disk_io

result = [[experiment(sliding_number=i) for i in np.arange(2,6)] for j in np.arange(2)]
cols = ["sliding_number"]
cols.extend(metrics)
results = pd.DataFrame(np.array(result).reshape(-1,len(cols)), columns=cols)
#results.to_csv('experiment_logs/fgabpnnm_experiment.csv')
results.to_csv('experiment_logs/fuzzy_bpnn_experimentm.csv')

