import keras
from keras.layers import Dense, LSTM, Bidirectional, Dropout


class BiLSTM:
    def __init__(self, lstm_output, MODEL_MAX_LENGTH, EMBEDDING_SIZE, n_classes, model_name, **kwargs):
        self.model_name = model_name
        self.lstm_output = lstm_output
        self.MODEL_MAX_LENGTH = MODEL_MAX_LENGTH
        self.EMBEDDING_SIZE = EMBEDDING_SIZE
        self.dp_lstm = kwargs.get('dp_lstm', 0.2)
        self.dp_layer_1 = kwargs.get('dp_layer_1', 0.1)
        self.dp_layer_2 = kwargs.get('dp_layer_2', 0.1)
        self.n_classes = n_classes
        self.model = None
        self.save_dir = kwargs.get('save_dir','/content')

    def build_model(self):
        self.model = keras.Sequential()

        self.model.add(Bidirectional(LSTM(self.lstm_output, dropout=self.dp_lstm), input_shape=(self.MODEL_MAX_LENGTH,self.EMBEDDING_SIZE)))
        self.model.add(Dense(128, activation = 'relu'))
        if self.dp_layer_1: self.model.add(Dropout(self.dp_layer_1))
        self.model.add(Dense(64, activation = 'relu'))
        if self.dp_layer_2: self.model.add(Dropout(self.dp_layer_2))
        self.model.add(Dense(self.n_classes, activation = 'softmax'))
        self.model.summary()

    def display_model_summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Build Model to print a summary!!!")
    
    def train(self, BATCH_SIZE, EPOCHS, X_train, Y_train, X_test, Y_test):
        self.model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        history = self.model.fit(X_train, Y_train, epochs = EPOCHS, batch_size=BATCH_SIZE, verbose = 1, validation_data =(X_test, Y_test))
        return history
    
    def save_model(self):
        model_dir = f"{self.save_dir}/{self.model_name}"
        self.model.save(model_dir)

    def get_model(self):
        return self.model