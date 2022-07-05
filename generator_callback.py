
from tensorflow.keras.callbacks import Callback


class GeneratorCallback(Callback):

    def __init__(self, generator, n_batch, model_file):
        self.generator = generator
        self.n_batch = n_batch
        self.model_file = model_file
        self.best_overall = 0

    def on_train_batch_end(self, batch, logs=None):

        loss_weight = 0.0
        accuracy_weight = 1.0

        if batch % self.n_batch == 0:
            loss, accuracy = self.model.evaluate(self.generator, batch_size=self.n_batch, verbose=0)
            print(' Validation Accuracy: %.2f' % accuracy)
            overall = (accuracy * accuracy_weight) + (loss_weight/loss)

            if overall > self.best_overall:
                self.best_overall = overall
                print('Saving best overall model...')
                self.model.save(self.model_file)