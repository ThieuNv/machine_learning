from __init__fuzzy import *

dat = pd.read_csv('sampling_617685_metric_10min_datetime_normalize.csv',parse_dates=True,index_col=0)[:3000]
gFeeder = GFeeder()
metrics = ['cpu_rate','mem_usage','disk_space']
def experiment(sliding_number):
    X_train, y_train, X_test, y_test = gFeeder.split_train_and_test(dat, metrics=metrics, n_sliding_window=sliding_number)
    n_hidden = 32
    fit_params = {
        'neural_shape': [len(X_train[0]), n_hidden, len(metrics)]
    }
    ga_estimator = GAEstimator(cross_rate=0.5, mutation_rate=0.06, gen_size=100, pop_size=30)
    nn = KerasRegressor(hidden_nodes=[n_hidden], optimize='adam'
                             , steps=7000, learning_rate=1E-02)
    classifier = OptimizerNNEstimator(ga_estimator, nn)
    a = classifier.fit(X_train, y_train, **fit_params)
    ypred = classifier.predict(X_test)
    y_cpu = ypred[:,0]
    y_ram = ypred[:,1]
    y_disk_space = ypred[:,2]
    score_mae_CPU = mean_absolute_error(y_cpu, y_test[:,0])
    score_mae_RAM =  mean_absolute_error(y_ram, y_test[:,1])
    score_mae_diskio = mean_absolute_error(diskio_scaler.inverse_transform(y_disk_space),diskio_scaler.inverse_transform(y_test[:,2]))
    np.savez('model_saved/GABPNNM_%s_%s'%(sliding_number,score_mae_CPU),y_pred=ypred, y_true=y_test)
    return [sliding_number, score_mae_CPU, score_mae_RAM, score_mae_diskio]

result = [[experiment(sliding_number=i) for i in np.arange(2,6)] for j in np.arange(3)]
cols = ["sliding_number"]
cols.extend(metrics)
results = pd.DataFrame(np.array(result).reshape(-1,len(cols)), columns=cols)
#results.to_csv('experiment_logs/fgabpnnm_experiment.csv')
results.to_csv('experiment_logs/gabpnnm_experimentm.csv')


