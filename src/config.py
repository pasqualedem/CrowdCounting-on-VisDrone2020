from easydict import EasyDict
import time

__C = EasyDict()
cfg = __C

__C.SEED = 3035  # random seed

# System settings
__C.TRAIN_BATCH_SIZE = 4
__C.VAL_BATCH_SIZE = 6
__C.N_WORKERS = 4

__C.PRE_TRAINED = 'exp/05-30_23-48_VisDrone_MobileCountx0_75_0.0001__540x960_SGD/all_ep_139_mae_7.1_rmse_9.4.pth'

# path settings
__C.EXP_PATH = './exp'
__C.DATASET = 'VisDrone'
__C.NET = 'MobileCountx0_75'
__C.DETAILS = '_540x960_SGD'

# learning optimizer settings
__C.LR = 1e-4  # learning rate
__C.W_DECAY = 1e-4  # weight decay
__C.LR_DECAY = 0.995  # decay rate
__C.LR_DECAY_START = 0  # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1  # decay frequency
__C.MAX_EPOCH = 500

__C.OPTIM_ADAM = ('Adam',
                  {
                      'lr': __C.LR,
                      'weight_decay': __C.W_DECAY,
                  })
__C.OPTIM_SGD = ('SGD',
                  {
                      'lr': __C.LR,
                      'weight_decay': __C.W_DECAY,
                      'momentum': 0.95
                  })

__C.OPTIM = __C.OPTIM_SGD  # Chosen optimizer

__C.PATIENCE = 40
__C.EARLY_STOP_DELTA = 1e-2

# print
__C.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())
__C.EXP_NAME = now \
               + '_' + __C.DATASET \
               + '_' + __C.NET \
               + '_' + str(__C.LR) \
               + '_' + __C.DETAILS
__C.DEVICE = 'cuda'  # cpu or cuda

# ------------------------------VAL------------------------
__C.VAL_SIZE = 0.2
__C.VAL_DENSE_START = 1
__C.VAL_FREQ = 10  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ
