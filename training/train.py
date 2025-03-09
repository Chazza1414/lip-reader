from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.callbacks import TensorBoard
from keras.metrics import CategoricalAccuracy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.generator import Generator, LockedIterator
from lipreader.common.constants import IMAGE_HEIGHT, IMAGE_WIDTH, VIDEO_FRAME_NUM, MODEL_SAVE_LOCATION, IMAGE_CHANNELS, MODEL_LOG_LOCATION, DATASET_PATH
from lipreader.model import LipReader
import numpy as np
import datetime
from pathlib import Path

np.random.seed(55)

def train(run_name, stop_epoch, img_c, img_w, img_h, frames_n, minibatch_size, dataset_path):

    lipreader_generator = Generator(minibatch_size=minibatch_size, dataset_path=dataset_path).build()

    lipreader = LipReader(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n, output_size=lipreader_generator.output_size)
    lipreader.summary()

    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    loss = CategoricalCrossentropy(reduction="sum_over_batch_size", name="CCE")
    accuracy = CategoricalAccuracy(name="categorical_accuracy")

    lipreader.model.compile(optimizer=adam, loss=loss, metrics=[accuracy])

    tensorboard = TensorBoard(log_dir=os.path.join(MODEL_LOG_LOCATION, run_name), histogram_freq=1, write_images=True, embeddings_freq=1, 
                              update_freq=10)
    
    print("training " + str(run_name))
    lipreader.model.fit(
        x=LockedIterator(lipreader_generator.next_train()),
        steps_per_epoch=lipreader_generator.default_training_steps, epochs=stop_epoch,
        validation_data=LockedIterator(lipreader_generator.next_validate()), validation_steps=lipreader_generator.default_validation_steps,
        callbacks=[tensorboard],
        verbose=1,
        max_queue_size=5,
        workers=2)
    
    #  to get rid of the warning from absl, save file as .h5
    lipreader.model.save(Path(MODEL_SAVE_LOCATION) / run_name)

    print("evaluating")
    lipreader.model.evaluate(
        x=LockedIterator(lipreader_generator.next_evaluate()), 
        verbose=1, callbacks=[tensorboard],
        max_queue_size=5,
        workers=2,
        steps=len(lipreader_generator.evaluate_list))

if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    #train(run_name, 1, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT, VIDEO_FRAME_NUM, 1, dataset_path="H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\test_datasets")
    train(run_name, 4, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT, VIDEO_FRAME_NUM, 16, dataset_path=DATASET_PATH)