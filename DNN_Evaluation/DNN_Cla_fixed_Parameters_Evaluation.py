import utils.evaluation
import DNN_Scorers.ACA_Scorers as aca
import time

if __name__ == '__main__':
    utils.evaluation.evaluate_DNN('Results_no_Folds_' + time.asctime(time.localtime()), '/home/tb/Documents/DNN/DNN_Cla_fixed_Parameters/final',
                                  meta_data_files=['DNN_Cla_fixed_Parameters_DefaultArchitecture_dataset_YAU038_drop_hidden_0_5_drop_visible_0_2_lr_0_001_optimizer_type_Adam_shuffle_True_folds_1_Sun_Feb_12_075713_2023.pkl'],
                                  scorers=[ aca.ACA_Layer_Scorer(), aca.ACA_Unit_Scorer()],
                                  plotters=[])

    #utils.evaluation.evaluate_DNN('Results_5_Folds_' + time.asctime(time.localtime()), '/home/tb/Documents/DNN/DNN_Cla_fixed_Parameters/final',
    #                              meta_data_files=['DNN_Cla_dataset_YAU038_drop_hidden_0_5_drop_visible_0_2_lr_0_001_optimizer_type_Adam_shuffle_True_folds_5_Fri_Feb__3_163506_2023.pkl'],
    #                              scorers=[utils.evaluation.F1_Scorer(), utils.evaluation.ROC_AUC_Scorer(), aca.ACA_Layer_Scorer(), aca.ACA_Unit_Scorer()])