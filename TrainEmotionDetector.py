import keras
import tensorflow as tf
Input = tf.keras.layers.Input
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Flatten = keras.layers.Flatten
Adam = keras.optimizers.Adam
Sequential = keras.models.Sequential
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
ExponentialDecay = keras.optimizers.schedules.ExponentialDecay


train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    'data1/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_generator = validation_data_gen.flow_from_directory(
    'data1/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)


lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)

emotion_model = Sequential()

emotion_model.add(Input(shape=(48, 48, 1)))
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())

emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(1000, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(1000, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(800, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(700, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(350, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(100, activation='relu'))
emotion_model.add(Dropout(0.5))

emotion_model.add(Dense(3, activation='softmax'))

emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=lr_schedule),
    metrics=['accuracy']
)

previous_accuracy = 0.0

for epoch in range(50):  
    print(f"Epoch {epoch+1}")
    emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        verbose=1
    )

    current_accuracy = emotion_model_info.history['accuracy'][-1]

    if current_accuracy < previous_accuracy:
        print(f"Accuracy decreased at epoch {epoch+1}, skipping this epoch's results.")
        continue
    
    if current_accuracy >= 0.9080:
        print(f"Accuracy goal of 0.9080 reached at epoch {epoch+1}. Stopping training.")
        break

    previous_accuracy = current_accuracy

model_json = emotion_model.to_json()  
with open("excel_model.json", "w") as json_file:
    json_file.write(model_json)

emotion_model.save_weights('excel_model.weights.h5')