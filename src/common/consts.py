CLASSES_COUNT = 50
IMG_PER_CLASS = 200
TRAIN_IMAGE_COUNT = CLASSES_COUNT * IMG_PER_CLASS

INCEPTION_CLASSES_COUNT = 2048
INCEPTION_OUTPUT_FIELD = 'inception_output'
LABEL_ONE_HOT_FIELD = 'label_one_hot'
IMAGE_RAW_FIELD = 'image_raw'
INCEPTION_INPUT_TENSOR = 'DecodeJpeg/contents:0'
INCEPTION_OUTPUT_TENSOR = 'pool_3:0'
OUTPUT_NODE_NAME = 'output_node'
OUTPUT_TENSOR_NAME = OUTPUT_NODE_NAME + ':0'
HEAD_INPUT_NODE_NAME = 'x'
HEAD_INPUT_TENSOR_NAME = HEAD_INPUT_NODE_NAME + ':0'

TRAIN_SAMPLE_SIZE = 1000
VALIDATION_IMAGE_COUNT = 1000
BATCH_SIZE = 64
EPOCHS_COUNT = 5000
LEARNING_RATE = 0.0001
NEURONS = 1024

NAME = "old_dataset_split"

# name of the model being referenced by all other scripts
CURRENT_MODEL_NAME = '{}_c{}_n{}_e{}_b{}_l{}'.format(NAME, CLASSES_COUNT, NEURONS, EPOCHS_COUNT, BATCH_SIZE, LEARNING_RATE)
# sets up number of layers and number of units in each layer for
# the "head" dense neural network stacked on top of the Inception
# pre-trained model.
HEAD_MODEL_LAYERS = [INCEPTION_CLASSES_COUNT, 1024, CLASSES_COUNT]