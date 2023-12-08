import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("train.head()")
print(train.head())

y_train = train['label']
X_train = train.drop(columns=['label']).copy()

print("X_train.shape", X_train.shape)
print("test.shape", test.shape)
print("sum(X_train.isnull().sum())", sum(X_train.isnull().sum()))
print("set(X_train.loc[0])", set(X_train.loc[0]))
print("set(y_train)", set(y_train))

sns.set(style='white', context='notebook', palette='Paired')
g = sns.countplot(x=y_train)
plt.show()

X_train /= 255
test /= 255

X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, num_classes = 10)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                            test_size=0.2, random_state=0)

# data augmentation
dataGen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=20,
        zoom_range = 0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False,
        vertical_flip=False)
dataGen.fit(X_train)

# model
model = Sequential()

model.add(Conv2D(filters = 128, kernel_size = (3, 3),padding = 'Same', 
                activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 128, kernel_size = (3, 3),padding = 'Same', 
                activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3, 3),padding = 'Same', 
                activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3),padding = 'Same', 
                activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 32, kernel_size = (3, 3),padding = 'Same', 
                activation ='relu'))
model.add(Conv2D(filters = 32, kernel_size = (3, 3),padding = 'Same', 
                activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(16, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

print(model.summary())

model.compile(loss='categorical_crossentropy', 
            metrics=["accuracy"],
            optimizer=Adam(learning_rate=0.0003))

epochs = 20
BS = 64

history = model.fit(dataGen.flow(X_train,y_train, batch_size = BS),
                    epochs= epochs,
                    validation_data=(X_valid, y_valid),
                    verbose = 1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

# predict
y_pred = model.predict(X_valid)
y_classes = np.argmax(y_pred,axis = 1) 

# acc
y_true = np.argmax(y_valid,axis = 1) 

confusion_mtx = confusion_matrix(y_true, y_classes) 
print(confusion_mtx)

classes = range(10)

plt.figure(figsize=(8, 6))
plt.imshow(confusion_mtx, interpolation='nearest', cmap = plt.cm.Reds)
plt.title('Confusion matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = confusion_mtx.max() / 2.
for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
    plt.text(j, i, confusion_mtx[i, j],
            horizontalalignment="center",
            color="white" if confusion_mtx[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

results = model.predict(test)
results_classes = np.argmax(results, axis = 1)

submission = pd.DataFrame({
    'ImageId': np.arange(1,len(test)+1),
    'Label': results_classes
})

submission.to_csv("data/submission.csv",index=False)