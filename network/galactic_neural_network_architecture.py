import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate

class GalacticNeuralNetworkArchitecture:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_model(self):
        # Input layers
        distance_input = Input(shape=(self.input_shape[0],), name='distance_input')
        velocity_input = Input(shape=(self.input_shape[1],), name='velocity_input')
        galaxy_type_input = Input(shape=(self.input_shape[2],), name='galaxy_type_input')

        # Distance branch
        x_distance = Dense(64, activation='relu')(distance_input)
        x_distance = Dropout(0.2)(x_distance)
        x_distance = Dense(32, activation='relu')(x_distance)

        # Velocity branch
        x_velocity = Dense(64, activation='relu')(velocity_input)
        x_velocity = Dropout(0.2)(x_velocity)
        x_velocity = Dense(32, activation='relu')(x_velocity)

        # Galaxy type branch
        x_galaxy_type = Dense(64, activation='relu')(galaxy_type_input)
        x_galaxy_type = Dropout(0.2)(x_galaxy_type)
        x_galaxy_type = Dense(32, activation='relu')(x_galaxy_type)

        # Merge branches
        x = concatenate([x_distance, x_velocity, x_galaxy_type])

        # Output layer
        x = Dense(self.num_classes, activation='softmax')(x)

        # Create model
        model = Model(inputs=[distance_input, velocity_input, galaxy_type_input], outputs=x)

        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

def main():
    input_shape = (1, 1, 1)  # distance, velocity, galaxy type
    num_classes = 3  # Spiral, Elliptical, Irregular
    gnn = GalacticNeuralNetworkArchitecture(input_shape, num_classes)
    model = gnn.create_model()
    print(model.summary())

if __name__ == "__main__":
    main()
