from __init__fuzzy import *
from joblib import Parallel, delayed
dat = pd.read_csv('sampling_617685_metric_10min_datetime_normalize.csv',parse_dates=True,index_col=0)[:3000]
gFeeder = GFeeder()
metrics = ['cpu_rate','mem_usage','disk_space']
def experiment(sliding_number):
    X_train, y_train, X_test, y_test = gFeeder.split_train_and_test(dat, metrics=metrics, n_sliding_window=sliding_number)
    classifier = KerasRegressor(hidden_nodes=[32],steps=8000,learning_rate=1E-02,batch_size=64)
    a = classifier.fit(X_train, y_train)
    ypred = classifier.predict(X_test)
    y_cpu = ypred[:,0]
    y_ram = ypred[:,1]
    y_disk_space = ypred[:,2]
    score_mae_CPU = mean_absolute_error(cpu_scaler.inverse_transform(y_cpu), cpu_scaler.inverse_transform(y_test[:,0]))
    score_mae_RAM =  mean_absolute_error(ram_scaler.inverse_transform(y_ram), ram_scaler.inverse_transform(y_test[:,1]))
    score_mae_diskio = mean_absolute_error(diskio_scaler.inverse_transform(y_disk_space),diskio_scaler.inverse_transform(y_test[:,2]))
    np.savez('model_saved/BPNNM_%s_%s'%(sliding_number,score_mae_CPU),y_pred=ypred, y_true=y_test)
    return [sliding_number, score_mae_CPU, score_mae_RAM, score_mae_diskio]

result = [[experiment(sliding_number=i) for i in np.arange(2,6)] for j in np.arange(2)]
cols = ["sliding_number"]
cols.extend(metrics)
results = pd.DataFrame(np.array(result).reshape(-1,len(cols)), columns=cols)
results.to_csv('experiment_logs/bpnnm_experiment.csv')

