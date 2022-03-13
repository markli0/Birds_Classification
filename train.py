import os
import numpy as np
import tensorflow as tf
import Models.Resnets
from tqdm import tqdm, tqdm_notebook
import dataloader
import pandas as pd


def load_model(weight=None):
    model = Models.Resnets.resnet32()
    # print(model.summary())
    if weight:
        model.load_weights(os.path.join('saved_models', weight))

    return model


def train_model(model, run_name):
    logdir = 'logs'
    val_interval = 1200

    lr_values = [0.1, 0.01, 0.001, 0.0001]
    lr_boundaries = [60*val_interval, 120*val_interval, 160*val_interval, 200*val_interval]
    log_interval = 200
    batch_size = 32
    nesterov = False

    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundaries[:-1], values=lr_values)
    optimizer = tf.keras.optimizers.SGD(schedule, momentum=0.9, nesterov=nesterov)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    ds = dataloader.load_datasets()

    ds = ds.shuffle(10000).batch(batch_size).prefetch(-1)
    ds_train = ds.take(val_interval)
    ds_test = ds.skip(1100)

    runid = run_name
    writer = tf.summary.create_file_writer(logdir + '/' + runid)
    accuracy = tf.metrics.SparseCategoricalAccuracy()
    cls_loss = tf.metrics.Mean()
    reg_loss = tf.metrics.Mean()
    
    print(f"RUNID: {runid}")
    tf.keras.utils.plot_model(model, os.path.join('saved_plots', runid + '.png'))
    
    @tf.function
    def step(x, y, training):
        with tf.GradientTape() as tape:
            r_loss = tf.add_n(model.losses)
            outs = model(x, training)
            c_loss = loss_fn(y, outs)
            loss = c_loss + r_loss

        if training:
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        accuracy(y, outs)
        cls_loss(c_loss)
        reg_loss(r_loss)

    training_step = 0
    best_validation_acc = 0
    epochs = lr_boundaries[-1] // val_interval
    
    for epoch in range(epochs):
        for x, y in tqdm(ds_train.take(val_interval), desc=f'epoch {epoch+1}/{epochs}',
                         total=val_interval, ncols=100, ascii=True):

            training_step += 1
            step(x, y, training=True)

            if training_step % log_interval == 0:
                with writer.as_default():
                    c_loss, r_loss, err = cls_loss.result(), reg_loss.result(), 1-accuracy.result()
                    print(f" c_loss: {c_loss:^6.3f} | r_loss: {r_loss:^6.3f} | err: {err:^6.3f}", end='\r')

                    tf.summary.scalar('train/error_rate', err, training_step)
                    tf.summary.scalar('train/classification_loss', c_loss, training_step)
                    tf.summary.scalar('train/regularization_loss', r_loss, training_step)
                    tf.summary.scalar('train/learnig_rate', optimizer._decayed_lr('float32'), training_step)
                    cls_loss.reset_states()
                    reg_loss.reset_states()
                    accuracy.reset_states()

        for x, y in ds_test:
            step(x, y, training=False)

        with writer.as_default():
            tf.summary.scalar('test/classification_loss', cls_loss.result(), step=training_step)
            tf.summary.scalar('test/error_rate', 1-accuracy.result(), step=training_step)
            
            if accuracy.result() > best_validation_acc:
                best_validation_acc = accuracy.result()
                model.save_weights(os.path.join('saved_models', runid + '.tf'))
                
            cls_loss.reset_states()
            accuracy.reset_states()


def test_model(model=None):
    model.compile()
    paths, imgs = dataloader.load_test_ds()
    maps = dataloader.load_maps()

    labels = []
    for img in imgs:
        ds = tf.data.Dataset.from_tensor_slices([img])
        ds = ds.prefetch(-1)
        label = np.argmax(model.predict(ds), axis=1)
        label = maps[label[0]]
        labels.append(label)
        #print(label)

    data = {'path': paths,
            'class': labels}
    df = pd.DataFrame(data)

    df.to_csv('output.csv')


def most_frequent(List):
    return max(set(List), key = List.count)


def rematch_classes(model):
    model.compile()

    maps = [];
    for i in range(555):
        paths, imgs = dataloader.load_train_ds(i)

        labels = []
        for img in imgs:
            ds = tf.data.Dataset.from_tensor_slices([img])
            ds = ds.prefetch(-1)
            label = np.argmax(model.predict(ds), axis=1)
            labels.append(label[0])

        new_label = most_frequent(labels)
        maps.append(new_label)

    data = {'maps': maps}
    df = pd.DataFrame(data)

    df.to_csv('map.csv')



# print(device_lib.list_local_devices())
train_model(load_model(), 'resnet32-0311')
# rematch_classes(load_model('resnet32-0309.tf'))
# test_model(load_model('resnet32-0309.tf'))

