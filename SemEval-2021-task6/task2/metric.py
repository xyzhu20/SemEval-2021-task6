from sklearn.metrics import f1_score
import tensorflow as tf
from keras import backend as K

class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = self.model.predict(self.validation_data[0])
        # val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        threshold, upper, lower = 0.5, 1, 0
        val_predict[val_predict > threshold] = upper
        val_predict[val_predict <= threshold] = lower

        val_targ = self.validation_data[1]

        _val_f1_macro = f1_score(val_targ, val_predict, average='macro')
        _val_f1_micro = f1_score(val_targ, val_predict, average='micro')
        # _val_recall = recall_score(val_targ, val_predict, average='macro')
        # _val_precision = precision_score(val_targ, val_predict, average='macro')
        logs['val_f1_macro'] = _val_f1_macro
        logs['val_f1_micro'] = _val_f1_micro

        # logs['val_recall'] = _val_recall
        # logs['val_precision'] = _val_precision
        print(" — val_f1_macro: %f  — val_f1_Micro: %f" % (_val_f1_macro, _val_f1_micro))
        return


def f1(y_true, y_pred):
    # f1值作为评估参数
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))