import utils.evaluation
import DNN_Scorers.ACA_Scorers as aca
import time

if __name__ == '__main__':
    utils.evaluation.evaluate_DNN('Results_no_Folds_' + time.asctime(time.localtime()), '/home/tb/Documents/DNN/DNN_Cla_Reg',
                                 meta_data_files=[
                                     'DNN_Cla_Reg_EnlargedEnvelopeLayerAfterFirstLinearLayerArchitecture_dataset_YAU038_drop_hidden_0_5_drop_visible_0_2_lr_0_001_optimizer_type_Adam_shuffle_True_folds_1_Fri_Mar_10_202618_2023.pkl'
                                 ],
                                 scorers=[ aca.ACA_Layer_Scorer(), aca.ACA_Unit_Scorer()],
                                 plotters=[])

    #utils.evaluation.evaluate_DNN('Results_5_Folds_' + time.asctime(time.localtime()),
    #                              '/home/tb/Documents/DNN/DNN_Cla_Reg',
    #                              meta_data_files=[
    #                                  'DNN_Cla_Reg_EnvelopeLayerAfterSecondLinearLayerArchitecture_dataset_YAU038_drop_hidden_0_5_drop_visible_0_2_lr_0_001_optimizer_type_Adam_shuffle_False_folds_5_Thu_Mar__9_175521_2023.pkl',
    #                                  'DNN_Cla_Reg_EnvelopeLayerAfterSecondLinearLayerArchitecture_dataset_YAU038_drop_hidden_0_5_drop_visible_0_2_lr_0_001_optimizer_type_Adam_shuffle_True_folds_5_Thu_Mar__9_175537_2023.pkl',
    #                                  'DNN_Cla_Reg_EnvelopeLayerAfterFirstLinearLayerArchitecture_dataset_YAU038_drop_hidden_0_5_drop_visible_0_2_lr_0_001_optimizer_type_Adam_shuffle_False_folds_5_Thu_Mar__9_174315_2023.pkl',
    #                                  'DNN_Cla_Reg_EnvelopeLayerAfterFirstLinearLayerArchitecture_dataset_YAU038_drop_hidden_0_5_drop_visible_0_2_lr_0_001_optimizer_type_Adam_shuffle_True_folds_5_Thu_Mar__9_174331_2023.pkl',
    #                                  'DNN_Cla_Reg_EnlargedEnvelopeLayerAfterFirstLinearLayerArchitecture_dataset_YAU038_drop_hidden_0_5_drop_visible_0_2_lr_0_001_optimizer_type_Adam_shuffle_False_folds_5_Thu_Mar__9_174907_2023.pkl',
    #                                  'DNN_Cla_Reg_EnlargedEnvelopeLayerAfterFirstLinearLayerArchitecture_dataset_YAU038_drop_hidden_0_5_drop_visible_0_2_lr_0_001_optimizer_type_Adam_shuffle_True_folds_5_Thu_Mar__9_174925_2023.pkl'
    #                                  ],
    #                              scorers=[utils.evaluation.F1_Scorer(), utils.evaluation.ROC_AUC_Scorer(),
    #                                       utils.evaluation.R2_Scorer()],
    #                              plotters=[])