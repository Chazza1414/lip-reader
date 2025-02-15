from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.generator import Generator
#from lipnet.lipreading.callbacks import Statistics, Visualize
#from lipnet.lipreading.curriculums import Curriculum
#from lipnet.core.decoders import Decoder
from lipreader.helpers import labels_to_text
from lipreader.spell import Spell
from lipreader.model import LipReader
import numpy as np
import datetime


np.random.seed(55)

#CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
#DATASET_DIR  = os.path.join(CURRENT_PATH, 'datasets')
#OUTPUT_DIR   = os.path.join(CURRENT_PATH, 'results')
#LOG_DIR      = os.path.join(CURRENT_PATH, 'logs')

# PREDICT_GREEDY      = False
# PREDICT_BEAM_WIDTH  = 200
# PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','..','common','dictionaries','grid.txt')

def train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size):
    #curriculum = Curriculum(curriculum_rules)
    lipreader_generator = Generator(minibatch_size=minibatch_size,
                                img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                                absolute_max_string_len=absolute_max_string_len,
                                start_epoch=start_epoch).build()

    lipreader = LipReader(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                            absolute_max_string_len=absolute_max_string_len, output_size=lipreader_generator.output_size)
    lipreader.summary()

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    lipreader.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

    # load weight if necessary
    # if start_epoch > 0:
    #     weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
    #     lipreader.load_weights(weight_file)

    # spell = Spell(path=PREDICT_DICTIONARY)
    # decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
    #                   postprocessors=[labels_to_text, spell.sentence])

    # # define callbacks
    # statistics  = Statistics(lipnet, lipreader_generator.next_val(), decoder, 256, output_dir=os.path.join(OUTPUT_DIR, run_name))
    # visualize   = Visualize(os.path.join(OUTPUT_DIR, run_name), lipnet, lipreader_generator.next_val(), decoder, num_display_sentences=minibatch_size)
    # tensorboard = TensorBoard(log_dir=os.path.join(LOG_DIR, run_name))
    # csv_logger  = CSVLogger(os.path.join(LOG_DIR, "{}-{}.csv".format('training',run_name)), separator=',', append=True)
    # checkpoint  = ModelCheckpoint(os.path.join(OUTPUT_DIR, run_name, "weights{epoch:02d}.h5"), monitor='val_loss', save_weights_only=True, mode='auto', period=1)
    print("training started")
    lipreader.fit(x=lipreader_generator.next_train(),
                        steps_per_epoch=lipreader_generator.default_training_steps, epochs=stop_epoch,
                        validation_data=lipreader_generator.next_val(), validation_steps=lipreader_generator.default_validation_steps,
                        #callbacks=[checkpoint, statistics, visualize, lipreader_generator, tensorboard, csv_logger], 
                        initial_epoch=start_epoch, 
                        verbose=1,
                        max_queue_size=5,
                        workers=2)

if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    train(run_name, 0, 5000, 3, 100, 50, 75, 32, 50)