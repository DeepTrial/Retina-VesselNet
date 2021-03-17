import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # use GPU-0

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import datetime
from util.load_cfg import train_cfg, dataset_cfg, sample_cfg
from util.dice import *
from model.unet import Unet
from data.train import dataloader

checkpoint_dir=train_cfg["checkpoint_dir"]
log_dir=train_cfg["log_dir"]

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

model=Unet(sample_cfg["patch_size"])

# Learning rate and optimizer （学习率调整和优化器）
cosine_decay = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=train_cfg["init_lr"], first_decay_steps=12000,t_mul=1000,m_mul=0.5,alpha=1e-5)
optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay)

# loss function （损失函数）
#loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)

# metric record （性能指标记录器）
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc=tf.keras.metrics.Mean(name='train_acc')
current_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_acc=tf.keras.metrics.Mean(name='val_acc')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

# checkpoint （模型存档管理器）
ckpt = tf.train.Checkpoint(model=model)
#ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))

# tensorboard writer （Tensorboard记录器）
log_dir=log_dir+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = tf.summary.create_file_writer(log_dir)


def train_step(step, patch, groundtruth):
    with tf.GradientTape() as tape:
        linear, pred_seg = model(patch, training=True)
        losses = dice_loss(groundtruth, pred_seg)

    # calculate the gradient （求梯度）
    grads = tape.gradient(losses, model.trainable_variables)
    # bp (反向传播)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # record the training loss and accuracy (记录loss和准确率)
    train_loss.update_state(losses)
    train_acc.update_state(dice(groundtruth, pred_seg))


def val_step(step, patch, groundtruth):
    linear, pred_seg = model(patch, training=False)
    losses = dice_loss(groundtruth, pred_seg)

    # record the val loss and accuracy (记录loss和准确率)
    val_loss.update_state(losses)
    val_acc.update_state(dice(groundtruth, pred_seg))

    tf.summary.image("image", patch, step=step)
    tf.summary.image("image transform", linear, step=step)
    tf.summary.image("groundtruth", groundtruth * 255, step=step)
    tf.summary.image("pred", pred_seg, step=step)
    log_writer.flush()

def train_function(train_dataset,val_dataset):
    lr_step = 0
    last_val_loss = 2e10
    EPOCHS=train_cfg["epoch"]
    VAL_TIME=train_cfg["val_time"]

    with log_writer.as_default():
        for epoch in range(EPOCHS):
            # renew the recorder （重置记录项）
            train_loss.reset_states()
            train_acc.reset_states()
            val_loss.reset_states()
            val_acc.reset_states()

            # training （训练部分）
            for tstep, (patch, groundtruth) in enumerate(train_dataset):
                train_step(lr_step, patch, groundtruth)

                tf.summary.scalar("learning_rate", optimizer._decayed_lr(tf.float32).numpy(), step=lr_step)
                print('\repoch {}, batch {}, loss:{:.4f}, dice:{:.4f}'.format(epoch + 1, tstep, train_loss.result(),train_acc.result()), end="")
                lr_step += 1

            if (epoch + 1) % VAL_TIME == 0:
                # valid (验证部分)
                for vstep, (patch, groundtruth) in enumerate(val_dataset):
                    val_step(lr_step, patch, groundtruth)

                print('\repoch {}, batch {}, train_loss:{:.4f}, train_dice:{:.4f}, val_loss:{:.4f}, val_dice:{:.4f}'.format(epoch + 1, vstep, train_loss.result(), train_acc.result(), val_loss.result(), val_acc.result()),end="")
                tf.summary.scalar("val_loss", val_loss.result(), step=epoch)
                tf.summary.scalar("val_dice", val_acc.result(), step=epoch)

                if val_loss.result() < last_val_loss:
                    ckpt.save(checkpoint_dir)
                    last_val_loss = val_loss.result()
            print("")
            tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
            tf.summary.scalar("train_dice", train_acc.result(), step=epoch)
            log_writer.flush()


if __name__ == "__main__":

    train_dataset,val_dataset=dataloader.get_dataset(dataset_cfg,regenerate=True)

    train_dataset = train_dataset.shuffle(buffer_size=1300).prefetch(train_cfg["batch_size"]).batch(train_cfg["batch_size"])
    val_dataset = val_dataset.shuffle(buffer_size=1300).prefetch(train_cfg["batch_size"]).batch(train_cfg["batch_size"])

    train_function(train_dataset,val_dataset)
