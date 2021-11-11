import tensorflow.keras.preprocessing as kp
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import tensorflow.keras.callbacks as callbacks

callback = callbacks.EarlyStopping(
    monitor = 'val_loss', min_delta = 0.001, patience = 3, mode = 'min'
)

train_generator = kp.image_dataset_from_directory(
 '/content/images',
 labels = "inferred",
 label_mode = 'categorical',
 class_names = ['food', 'interior', 'exterior'],
 color_mode="rgb",
 image_size=(300,300),
 batch_size=128,
 seed = 123,
 validation_split= 0.2,
 subset='training',
 interpolation = 'bilinear'
)

test_generator = kp.image_dataset_from_directory(
 '/content/images',
 labels="inferred",
 label_mode = 'categorical',
 color_mode="rgb",
 class_names = ['food', 'interior', 'exterior'],
 image_size=(300, 300),
 batch_size=128,
 seed = 123,
 validation_split=0.2,
 subset='validation',
 interpolation = 'bilinear')


batch_index = 0
test_batch = 0

model = Sequential([
        Input(shape=(300,300,3), name = 'input_layer'),
        BatchNormalization(),
        Conv2D(32, kernel_size=4,  strides= 1,activation='relu', name = 'conv_layer1'), 
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        Conv2D(64, kernel_size=3, strides=1, activation='relu', name = 'conv_layer2'),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        Conv2D(128, kernel_size=3, strides=1, activation='relu', name = 'conv_layer3'),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        Conv2D(256, kernel_size=3, strides=1, activation='relu', name = 'conv_layer4'),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(50, activation='relu', name = 'Dense_layer1'),
        Dense(100, activation='relu', name = 'Dense_layer2'),
        Dense(3, activation='softmax', name = 'output_layer') 
    ])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


history = model.fit(train_generator,batch_size=128, epochs = 10, validation_data=test_generator, validation_batch_size=128, callbacks = callback)
model.save('finalmodel')
def plot_loss_curve(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15,10))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper right')
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

plot_loss_curve(history.history)

#model = load_model('C:\\Users\\user\\Desktop\\3학년_2학기\\데이터\\model-201811250\\model-201811250')
test_generator = kp.image_dataset_from_directory(
 'C:\\Users\\user\\Desktop\\3학년_2학기\\데이터\\기말고사 대체과제\\images',
 labels="inferred",
 label_mode = 'categorical',
 color_mode="rgb",
 class_names = ['food', 'interior', 'exterior'],
 image_size=(300, 300),
 batch_size=128,
 seed = 123,
 validation_split=0.2,
 subset='validation',
 interpolation = 'bilinear')

predict = []
y = []
a = 1

for images, labels in test_generator:
  y_pred = model.predict(images,batch_size = 128)
  y_pred = np.argmax(y_pred, axis=1)
  labels = np.argmax(labels,axis=1)
  for i in range(len(y_pred)):
    #print(i)
    #plt.imshow(np.array(images[i]).astype('uint8'))
    #plt.show()
    print(labels[i] , " ", y_pred[i])
    y.append(labels[i])
    predict.append(y_pred[i])
    if labels[i] != y_pred[i]:
        print("sample %d is wrong!" % a)
        #plt.imshow(np.array(images[i]).astype('uint8'))
        #plt.show()
        #print("pause")
        with open("wrong_samples_fourteenth.txt", "a") as errfile:
            print("%d" % a, file=errfile)
    else:
        print("sample %d is correct" % a)
        #plt.imshow(np.array(images[i]).astype('uint8'))
        #plt.show()
        #print("pause")
    a += 1

print(classification_report(y, predict, target_names=['food','interior','exterior']))


