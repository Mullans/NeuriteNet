import tensorflow as tf
import tqdm.auto as tqdm

from .architecture import neurite_classifier
from .data import prepare_dataset


def train_wt_vs_cko(train_image_dir='TrainImages', validation_image_dir='ValidationImages'):
    train_data = prepare_dataset(base_image_dir=train_image_dir, batch_size=32, is_replate=False, is_training=True)
    validation_data = prepare_dataset(base_image_dir=validation_image_dir, batch_size=32, is_replate=False, is_training=False)
    model = neurite_classifier(out_classes=2, out_layers=1, dense_size=128)
    train_model(model,
                train_data,
                validation_data,
                epochs=300,
                num_classes=2,
                training_log='WTvsCKO_train_log',
                validation_log='WTvsCKO_val_log',
                model_save_path='WTvsCKO_model.tf')


def train_wt_vs_replate(train_image_dir='ReplateTrainImages', validation_image_dir='ReplateValidationImages'):
    train_data = prepare_dataset(base_image_dir=train_image_dir, batch_size=16, is_replate=True, is_training=True)
    validation_data = prepare_dataset(base_image_dir=validation_image_dir, batch_size=16, is_replate=True, is_training=False)
    model = neurite_classifier(out_classes=1, out_layers=2, dense_size=64)
    train_model(model,
                train_data,
                validation_data,
                epochs=50,
                num_classes=1,
                training_log='WTvsReplate_train_log',
                validation_log='WTvsReplate_val_log',
                model_save_path='WTvsReplate_model.tf')


def train_model(model, train_data, validation_data=None, learning_rate=1e-4, epochs=300, num_classes=2, training_log='training_log', validation_log='validation_log', model_save_path='model_weights.tf'):
    """Train the model.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The model to train
    train_data : tf.data.Dataset
        The training data
    validation_data : tf.data.Dataset | None
        Optional validation data
    learning_rate : float
        The learning rate
    epochs : int
        The total number of epochs to train for
    num_classes : int
        The number of predicted classes
    training_log : str
        The output location for the training tensorboard log file
    validation_log : str
        The output location for the validation tensorboard log file
    model_save_path : str
        The output location for the trained model
    """
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1)

    train_writer = tf.summary.create_file_writer(training_log)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_acc = [tf.keras.metrics.BinaryAccuracy('train_accuracy_{}'.format(idx)) for idx in range(num_classes)]

    if validation_data is not None:
        val_writer = tf.summary.create_file_writer(validation_log)
        val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        val_acc = [tf.keras.metrics.BinaryAccuracy('val_accuracy_{}'.format(idx)) for idx in range(num_classes)]

    @tf.function
    def train_step(model, optimizer, x, y):
        with tf.GradientTape() as tape:
            predicted = model(x, training=True)
            loss = loss_func(y, predicted)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        train_loss(loss)
        y_split = tf.split(y, num_classes, axis=-1)
        pred_split = tf.split(predicted, num_classes, axis=-1)
        for acc, y, pred in zip(train_acc, y_split, pred_split):
            acc(y, pred)

    @tf.function
    def val_step(model, x, y):
        predicted = model(x, training=False)
        loss = loss_func(y, predicted)
        val_loss(loss)
        y_split = tf.split(y, num_classes, axis=-1)
        pred_split = tf.split(predicted, num_classes, axis=-1)
        for acc, y, pred in zip(val_acc, y_split, pred_split):
            acc(y, pred)

    pbar = tqdm.tqdm(total=epochs, unit=' epochs')
    for epoch in range(epochs):
        for image, label in train_data:
            train_step(model, opt, image, label)
        with train_writer.as_default():
            logstring = 'Epoch {:04d}'.format(epoch)
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            logstring += ', Loss: {:.4f}'.format(train_loss.result())
            accuracies = []
            for i, acc in enumerate(train_acc):
                tf.summary.scalar('accuracy_{}'.format(i), acc.result(), step=epoch)
                accuracies.append("{:.2f}".format(acc.result() * 100))
            logstring += ', Accuracy: ' + '/'.join(accuracies)
        train_loss.reset_states()
        for acc in train_acc:
            acc.reset_states()
        if validation_data is not None:
            logstring += ' || '
            for image, label in validation_data:
                val_step(model, image, label)
            with val_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=epoch)
                logstring += 'Val Loss: {:.4f}'.format(val_loss.result())
                accuracies = []
                for i, acc in enumerate(val_acc):
                    tf.summary.scalar('accuracy_{}'.format(i), acc.result(), step=epoch)
                    accuracies.append("{:.2f}".format(acc.result() * 100))
                logstring += ', Val Accuracy: ' + '/'.join(accuracies)
            val_loss.reset_states()
            for acc in val_acc:
                acc.reset_states()
        pbar.write(logstring)
        pbar.update(1)
    pbar.close()
    model.save(model_save_path)
