import threading
import DNN_Models.DNN_Cla_Reg as dnnMod
import utils.data
import utils.evaluation
from sklearn.model_selection import ParameterGrid
import time
import DNN_Scorers.ACA_Scorers as aca

if __name__ == "__main__":
    start_time= time.time()
    path_base= '/home/tb/Documents/DNN/DNN_Cla_Reg'
    model_base_name = 'DNN_Cla_Reg'
    model_class = dnnMod.cla_reg_model
    model_architecture_class = dnnMod.EnlargedEnvelopeLayerAfterFirstLinearLayerArchitecture
    dataset = ['YAU038']
    batch_sz = 512
    epochs = 1000
    lr_range=[0.001]
    drop_visible_range=[0.2]
    drop_hidden_range=[0.5]
    optimizer_type=['Adam']
    shuffle = [True]
    folds = 1
    gpuPreference = [0]
    evaluate = True

    grid=list(ParameterGrid({'dataset':dataset,
                             'drop_hidden':drop_hidden_range,
                             'drop_visible':drop_visible_range,
                             'lr':lr_range,
                             'optimizer_type':optimizer_type,
                             'shuffle':shuffle}))
    keys=list(grid[0].keys())
    values=[]
    maxRuns= len(lr_range) * len(drop_visible_range) * len(drop_hidden_range) * len(optimizer_type) * len(dataset) * len(shuffle)
    nGpu=gpuPreference.__len__()
    maxRunsOnGpu=1
    countRun=0
    countRunGpu=0
    gpuUsed=0
    pool_sema = threading.BoundedSemaphore(maxRunsOnGpu*nGpu)

    evaluate_meta_data_list = []
    threads = []
    while(gpuUsed < nGpu):
        while (countRunGpu < maxRunsOnGpu):
            if countRun<maxRuns:
                name = model_base_name + '_' + model_architecture_class.__name__
                for k in keys:
                    values.append(grid[countRun].get(k))
                    name = name + '_' + k + '_' + str(grid[countRun].get(k))
                name = name.replace('.', '_') + '_folds_' + str(folds) + '_' + time.asctime(time.localtime()).replace(' ', '_').replace(':', '')
                evaluate_meta_data_list.append(name + '.pkl')
                train_dataset = utils.data.SimpleDataset(dataset=values[0])
                X1, y1, Z1, *rest = train_dataset[0]
                model = model_class(model_architecture_class)
                model.model_creation({'n_features': len(X1), 'drop_visible': values[2],'drop_hidden': values[1], 'size_secondary_target': len(Z1)})
                thread = threading.Thread(target = model.train_DNN,
                                          args=(train_dataset,),
                                          kwargs=dict(sema=pool_sema, save_path=path_base, name=name,
                                                      optim_class=values[4], lr=values[3],
                                                      device_id=gpuPreference[gpuUsed], batch_sz=batch_sz,
                                                      epochs=epochs, folds=folds, shuffle=values[5], verbose=True))
                threads.append(thread)
                thread.start()
                print("thread " + name + ' started')
                values=[]
                countRun += 1
            countRunGpu += 1
        gpuUsed += 1
        countRunGpu=0
        if gpuUsed==nGpu and countRun<maxRuns:
            gpuUsed =0

    duration = time.time() - start_time
    print(duration)

    if evaluate:
        for t in threads:
            t.join()

        utils.evaluation.evaluate_DNN('Results_' + str(folds) + '_Folds_'+ time.asctime(time.localtime()),
                                      path_base,
                                      meta_data_files= evaluate_meta_data_list,
                                      scorers=[aca.ACA_Layer_Scorer(), aca.ACA_Unit_Scorer()],
                                      plotters=[])