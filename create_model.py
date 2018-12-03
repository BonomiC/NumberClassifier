import tensorflow as tf

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        tf.keras.layers.Dropout(0.3),
        # tf.keras.layers.Dense(512, activation=tf.nn.relu),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout

# def create_model():
# 	model = Sequential()
# 	model.add(Dense(512, activation='relu', input_dim=784))
# 	model.add(Dropout(0.5))
# 	model.add(Dense(512, activation='relu'))
# 	model.add(Dropout(0.5))
# 	model.add(Dense(10, activation='softmax'))

# 	adam = keras.optimizers.Adam(lr=0.001)

# 	model.compile(loss='categorical_crossentropy',
# 				optimizer=adam,
# 				metrics=['accuracy'])

# 	return model