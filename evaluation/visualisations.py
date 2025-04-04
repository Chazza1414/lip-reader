import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.generator import Generator, LockedIterator
from lipreader.videos import Video, VideoHelper
from lipreader.common.constants import MODEL_SAVE_LOCATION, NUM_PHONEMES, VIDEO_FRAME_NUM, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, PHONEME_LIST, DATASET_PATH, PHONEME_LIST
from pathlib import Path
from lipreader.align import Align
from keras import models


model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-03-12-10-09-28'
test_set_path = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\datasets\\evaluate"

testing_model = models.load_model(model_file_name)
videohelper = VideoHelper()
X_data_path = videohelper.enumerate_videos(test_set_path)
align_path = Path(DATASET_PATH) / 'phoneme-alignment'

lipreader_generator = Generator(minibatch_size=16, dataset_path=test_set_path).build()

#X_data_path = self.enumerate_videos(self.test_set_path)
X_data = []
Y_data = []

for path in X_data_path:
    print(path)
    video = Video().from_path(path)

    video_id = os.path.splitext(path)[0].split('\\')[-1]
    phoneme_alignment_path = os.path.join(align_path, video_id)+".txt"

    if (video is not None):
        X_data.append(video.frames)
        Y_data.append(Align(phoneme_alignment_path).alignment_matrix)

X_data = np.array(X_data).astype(np.float32) / 255

#predictions = testing_model.predict(X_data)
predictions = testing_model.predict(x=LockedIterator(lipreader_generator.next_evaluate()))
#predictions = testing_model.predict_on_batch(X_data)
# print(predictions)
# print(predictions.shape)

y_pred = []
y_true = []
labels_used = set()

for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        true = PHONEME_LIST[np.argmax(Y_data[i][j])]
        pred = PHONEME_LIST[np.argmax(predictions[i][j])]

        if (true == '*' or pred == '*'):
            continue

        y_pred.append(pred)
        y_true.append(true)
        labels_used.add(pred)
        labels_used.add(true)

        print(true, pred)

# print(y_pred)
# print(y_true)



# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, normalize='true')
#, labels=list(labels_used)
#, normalize='all'
# Print classification report
print(classification_report(y_true, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
#fmt="d",
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=np.unique(y_true), linewidths=1, yticklabels=np.unique(y_true))
#sns.heatmap(cm, annot=False, cmap="Blues", linewidths=1, xticklabels='auto', yticklabels='auto')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
