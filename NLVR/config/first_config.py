'''
Institution: Tulane University
Name: Chen Zheng
Date: 10/23/2018
Purpose: It is the global config file.
'''

CONFIG = {
    'hidden_size': 128,
    'embed_size': 64,
    'batch_size': 1,

    'feature_length': 10,
    'MAX_LENGTH': 20,
    'SOS_token': 0,
    'EOS_token': 1,
    'n_iters': 100,
    'print_every': 10,
    'plot_every': 100,
    'learning_rate': 0.01,
    'TRAIN_DIR': '../data/tmp.json',
    'TEST_DIR': '../data/test.json',
    'save_checkpoint_dir': '../check_point/2.pt',
    'save_train_result_dir': '../result/train/result.txt',
    'save_test_result_dir': '../result/test/result.txt',
    'TENSORBOARD_DIR': '../tensorboard/1.json'
}
