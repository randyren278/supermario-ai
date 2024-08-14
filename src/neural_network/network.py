from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

def build_neural_network(state_size, action_space, learning_rate):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=state_size, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

# Example usage
if __name__ == "__main__":
    model = build_neural_network((84, 84, 4), 3, 0.001)
    print(model.summary())
