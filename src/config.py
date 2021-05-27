from easydict import EasyDict
import time

__C = EasyDict()
cfg = __C

__C.SEED = 3035  # random seed


# System settings
__C.TRAIN_BATCH_SIZE = 1
__C.VAL_BATCH_SIZE = 6
__C.N_WORKERS = 4

__C.PRE_TRAINED = '../src/exp/05-25_19-52_VisDrone_MobileCount_0.0001/all_ep_90_mae_6.2_mse_8.4.pth'
__C.EXP_PATH = './exp'
__C.DATASET = 'VisDrone'
__C.NET = 'MobileCountx2'

# learning rate settings
__C.LR = 1e-4  # learning rate
__C.LR_DECAY = 0.995  # decay rate
__C.LR_DECAY_START = -1  # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1  # decay frequency
__C.MAX_EPOCH = 500
__C.INIT_EPOCH = 0

__C.PATIENCE = 15
__C.EARLY_STOP_DELTA = 1e-2

# print
__C.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())
__C.EXP_NAME = now \
               + '_' + __C.DATASET \
               + '_' + __C.NET \
               + '_' + str(__C.LR)
__C.DEVICE = 'cuda'  # cpu or cuda

# ------------------------------VAL------------------------
__C.VAL_SIZE = 0.2
__C.VAL_DENSE_START = 1
__C.VAL_FREQ = 10  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ
