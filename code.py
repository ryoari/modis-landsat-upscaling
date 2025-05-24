import tensorflow as tf
import os
import numpy as np
import rasterio 
import matplotlib.pyplot as plt
from glob import glob 
from google.colab import drive

print(f"TensorFlow Version: {tf.__version__}")


try:
    drive.mount('/content/drive', force_remount=True)
    DRIVE_PATH_PREFIX = "/content/drive/MyDrive/" 
    print("Google Drive mounted successfully.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    raise SystemExit("Google Drive mount failed. Cannot proceed.")


strategy = None
tpu_initialized_successfully = False
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print('Successfully connected to TPU: ', tpu.master())
    tpu_initialized_successfully = True
except Exception as e:
    print(f"TPU connection failed: {e}")
    print("Attempting to use GPU strategy...")
    try:
        gpus = tf.config.list_logical_devices('GPU')
        if gpus:
            strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
            print(f"Running on {len(gpus)} GPU(s).")
            for gpu_device in tf.config.list_physical_devices('GPU'):
                print(f"Found GPU: {gpu_device}")
        else:
            print("No GPUs found. Falling back to CPU strategy.")
            strategy = tf.distribute.get_strategy()
            print("Running on CPU.")
    except Exception as e_gpu:
        print(f"GPU strategy failed: {e_gpu}")
        strategy = tf.distribute.get_strategy()
        print("Running on CPU (final fallback).")
if strategy:
    print(f"Selected strategy: {strategy.__class__.__name__}")
    print("REPLICAS: ", strategy.num_replicas_in_sync)
else:
    print("CRITICAL ERROR: No strategy could be initialized.")
    raise SystemExit("Strategy initialization failed.")


IMG_HEIGHT = 716
IMG_WIDTH = 562
IMG_CHANNELS = 1
EPSILON = 1e-6
BATCH_SIZE_PER_REPLICA_GPU = 2
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA_GPU * strategy.num_replicas_in_sync
AUTO = tf.data.AUTOTUNE
EPOCHS = 75
L2_REG_FACTOR = 1e-5
UNET_DROPOUT_RATE = 0.25

print(f"Batch size per replica: {BATCH_SIZE_PER_REPLICA_GPU}")
print(f"Global batch size set to: {GLOBAL_BATCH_SIZE}")
DATA_DIR = os.path.join(DRIVE_PATH_PREFIX, 'DATA')
TRAIN_LR_DIR = os.path.join(DATA_DIR, 'train/lr')
TRAIN_HR_DIR = os.path.join(DATA_DIR, 'train/hr')
VAL_LR_DIR = os.path.join(DATA_DIR, 'val/lr')
VAL_HR_DIR = os.path.join(DATA_DIR, 'val/hr')
CHECKPOINT_DIR_BASE = 'unet_img_translation_gpu_v6_aug_reg'
CHECKPOINT_DIR = os.path.join(DRIVE_PATH_PREFIX, f'model_checkpoints/{CHECKPOINT_DIR_BASE}/')
try:
    tf.io.gfile.makedirs(CHECKPOINT_DIR)
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
except Exception as e:
    print(f"Error creating checkpoint directory: {e}")


def read_and_normalize_tiff(file_path_tensor):
    file_path = file_path_tensor.numpy().decode('utf-8')
    try:
        with rasterio.open(file_path) as src:
            img_masked_array = src.read(1, masked=True)
    except rasterio.errors.RasterioIOError as e:
        print(f"ERROR reading TIFF {file_path}: {e}. Returning zero array.")
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    img_data = img_masked_array.astype(np.float32)
    valid_pixels = img_data[~img_data.mask] if hasattr(img_data, 'mask') and np.any(img_data.mask) else img_data.flatten()
    if valid_pixels.size == 0:
        normalized_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    else:
        img_min = np.min(valid_pixels)
        img_max = np.max(valid_pixels)
        if (img_max - img_min) < EPSILON:
            normalized_img = np.zeros_like(img_data.data, dtype=np.float32)
        else:
            normalized_img = (img_data.data - img_min) / (img_max - img_min + EPSILON)
        if hasattr(img_data, 'mask') and np.any(img_data.mask):
             normalized_img[img_data.mask] = 0.0
    normalized_img = np.expand_dims(normalized_img, axis=-1) 
    return normalized_img.astype(np.float32)

def augment_pair(lr_image, hr_image):
    if tf.random.uniform(()) > 0.5:
        lr_image = tf.image.flip_left_right(lr_image)
        hr_image = tf.image.flip_left_right(hr_image)
    if tf.random.uniform(()) > 0.5:
        lr_image = tf.image.flip_up_down(lr_image)
        hr_image = tf.image.flip_up_down(hr_image)
    k_rot = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    lr_image = tf.image.rot90(lr_image, k=k_rot) 
    hr_image = tf.image.rot90(hr_image, k=k_rot)
    return lr_image, hr_image

@tf.function
def tf_load_and_preprocess_image_pair(lr_path, hr_path, augment=False):
    lr_image = tf.py_function(read_and_normalize_tiff, [lr_path], tf.float32)
    hr_image = tf.py_function(read_and_normalize_tiff, [hr_path], tf.float32)

    lr_image.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    hr_image.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])

    if augment:
        lr_image_aug, hr_image_aug = augment_pair(lr_image, hr_image)

        lr_image_res = tf.image.resize(lr_image_aug, [IMG_HEIGHT, IMG_WIDTH],
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        hr_image_res = tf.image.resize(hr_image_aug, [IMG_HEIGHT, IMG_WIDTH],
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        lr_image = lr_image_res
        hr_image = hr_image_res

        if tf.rank(lr_image) == 2:
            lr_image = tf.expand_dims(lr_image, axis=-1)
        if tf.rank(hr_image) == 2:
            hr_image = tf.expand_dims(hr_image, axis=-1)

    lr_image.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    hr_image.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])

    return lr_image, hr_image

try:
    train_lr_files = sorted([os.path.join(TRAIN_LR_DIR, f) for f in tf.io.gfile.listdir(TRAIN_LR_DIR) if f.endswith(('.tif', '.tiff'))])
    train_hr_files = sorted([os.path.join(TRAIN_HR_DIR, f) for f in tf.io.gfile.listdir(TRAIN_HR_DIR) if f.endswith(('.tif', '.tiff'))])
    val_lr_files = sorted([os.path.join(VAL_LR_DIR, f) for f in tf.io.gfile.listdir(VAL_LR_DIR) if f.endswith(('.tif', '.tiff'))])
    val_hr_files = sorted([os.path.join(VAL_HR_DIR, f) for f in tf.io.gfile.listdir(VAL_HR_DIR) if f.endswith(('.tif', '.tiff'))])
    print(f"Found {len(train_lr_files)} training LR, {len(train_hr_files)} training HR images.")
    print(f"Found {len(val_lr_files)} validation LR, {len(val_hr_files)} validation HR images.")
    assert len(train_lr_files) > 0 and len(train_hr_files) > 0, "Training files not found."
    assert len(train_lr_files) == len(train_hr_files), "Mismatch in training LR/HR file counts."
    if len(val_lr_files) > 0 or len(val_hr_files) > 0:
        assert len(val_lr_files) == len(val_hr_files), "Mismatch in validation LR/HR file counts."
    else: print("Warning: No validation files found.")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_lr_files, train_hr_files))
    train_dataset = train_dataset.shuffle(buffer_size=max(1, len(train_lr_files)))
    train_dataset = train_dataset.map(lambda lr, hr: tf_load_and_preprocess_image_pair(lr, hr, augment=True), num_parallel_calls=AUTO)
    train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTO)
    print("Training Dataset:", train_dataset)

    if len(val_lr_files) > 0:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_lr_files, val_hr_files))
        val_dataset = val_dataset.map(lambda lr, hr: tf_load_and_preprocess_image_pair(lr, hr, augment=False), num_parallel_calls=AUTO)
        val_dataset = val_dataset.batch(GLOBAL_BATCH_SIZE)
        val_dataset = val_dataset.prefetch(buffer_size=AUTO)
        print("Validation Dataset:", val_dataset)
    else:
        val_dataset = None
        print("No validation dataset will be used for training.")
except Exception as e:
    print(f"Error creating dataset pipelines: {e}")
    raise SystemExit("Dataset pipeline creation failed.")


def conv_block(input_tensor, num_filters, kernel_size=(3, 3), dropout_rate=0.0):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same",
                               kernel_regularizer=tf.keras.regularizers.l2(L2_REG_FACTOR))(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    if dropout_rate > 0: x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same",
                               kernel_regularizer=tf.keras.regularizers.l2(L2_REG_FACTOR))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def encoder_block(input_tensor, num_filters, dropout_rate=0.0):
    x = conv_block(input_tensor, num_filters, dropout_rate=dropout_rate)
    p = tf.keras.layers.MaxPooling2D((2, 2))(x)
    return x, p

def crop_and_concat_fn(inputs_list):
    upsampled, skip_connection = inputs_list[0], inputs_list[1]
    target_h, target_w = tf.shape(upsampled)[1], tf.shape(upsampled)[2]
    skip_h, skip_w = tf.shape(skip_connection)[1], tf.shape(skip_connection)[2]
    crop_h_total, crop_w_total = tf.maximum(0, skip_h - target_h), tf.maximum(0, skip_w - target_w)
    top_crop, left_crop = crop_h_total // 2, crop_w_total // 2
    bottom_crop, right_crop = crop_h_total - top_crop, crop_w_total - left_crop
    cropping_needed = tf.logical_or(tf.greater(crop_h_total, 0), tf.greater(crop_w_total, 0))
    def perform_crop(): return tf.slice(skip_connection, [0, top_crop, left_crop, 0], [-1, target_h, target_w, -1])
    def no_crop(): return skip_connection
    cropped_skip_features = tf.cond(cropping_needed, perform_crop, no_crop)
    return tf.keras.layers.Concatenate(axis=-1)([upsampled, cropped_skip_features])

def decoder_block(input_tensor, skip_features, num_filters, dropout_rate=0.0):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same",
                                        kernel_regularizer=tf.keras.regularizers.l2(L2_REG_FACTOR))(input_tensor)
    def get_lambda_output_shape(input_shapes_list):
        upsampled_shape, skip_shape = input_shapes_list[0], input_shapes_list[1]
        return (upsampled_shape[0], upsampled_shape[1], upsampled_shape[2], num_filters + skip_shape[3])
    x = tf.keras.layers.Lambda(crop_and_concat_fn, output_shape=get_lambda_output_shape)([x, skip_features])
    x = conv_block(x, num_filters, dropout_rate=dropout_rate)
    return x

def build_unet_image_translation(input_shape, num_classes=1, output_activation='sigmoid',
                                 dropout_rate_encoder_decoder=0.2, dropout_rate_bottleneck=0.3):
    inputs = tf.keras.layers.Input(input_shape)
    target_height, target_width = input_shape[0], input_shape[1]
    s1, p1 = encoder_block(inputs, 32, dropout_rate=dropout_rate_encoder_decoder)
    s2, p2 = encoder_block(p1, 64, dropout_rate=dropout_rate_encoder_decoder)
    s3, p3 = encoder_block(p2, 128, dropout_rate=dropout_rate_encoder_decoder)
    s4, p4 = encoder_block(p3, 256, dropout_rate=dropout_rate_encoder_decoder)
    b1 = conv_block(p4, 512, dropout_rate=dropout_rate_bottleneck)
    d1 = decoder_block(b1, s4, 256, dropout_rate=dropout_rate_encoder_decoder)
    d2 = decoder_block(d1, s3, 128, dropout_rate=dropout_rate_encoder_decoder)
    d3 = decoder_block(d2, s2, 64, dropout_rate=dropout_rate_encoder_decoder)
    d4 = decoder_block(d3, s1, 32, dropout_rate=dropout_rate_encoder_decoder)
    final_features = tf.keras.layers.Resizing(height=target_height,width=target_width,interpolation='bilinear')(d4)
    outputs = tf.keras.layers.Conv2D(num_classes, (1,1), padding="same", activation=output_activation,
                                     kernel_regularizer=tf.keras.regularizers.l2(L2_REG_FACTOR))(final_features)
    model = tf.keras.Model(inputs, outputs, name="UNet_Regularized")
    return model

try:
    with strategy.scope():
        model = build_unet_image_translation((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                                             num_classes=IMG_CHANNELS,
                                             output_activation='sigmoid',
                                             dropout_rate_encoder_decoder=UNET_DROPOUT_RATE,
                                             dropout_rate_bottleneck=min(UNET_DROPOUT_RATE + 0.1, 0.5))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=tf.keras.losses.MeanAbsoluteError())
    model.summary(line_length=120)
except Exception as e:
    print(f"Error building or compiling model: {e}")
    import traceback; traceback.print_exc()
    raise SystemExit("Model creation failed.")

checkpoint_filepath = os.path.join(CHECKPOINT_DIR, 'ckpt_epoch_{epoch:02d}.weights.h5')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True,
    monitor='val_loss' if val_dataset else 'loss', mode='min',
    save_best_only=False, save_freq='epoch')
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss' if val_dataset else 'loss', patience=15,
    restore_best_weights=True, verbose=1)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss' if val_dataset else 'loss', factor=0.2, patience=7,
    min_lr=1e-6, verbose=1)
callbacks_list = [model_checkpoint_callback, early_stopping_callback, reduce_lr_callback]

history = None
if len(train_lr_files) > 0 :
    print("Starting training with augmentation and regularization...")
    try:
        latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if latest_checkpoint:
            print(f"Loading weights from checkpoint: {latest_checkpoint}")
            model.load_weights(latest_checkpoint)
        else:
            print("No checkpoint found, starting training from scratch.")
        history = model.fit(
            train_dataset, epochs=EPOCHS, validation_data=val_dataset,
            callbacks=callbacks_list)
        print("Training finished.")
        final_weights_path = os.path.join(CHECKPOINT_DIR, 'final_model_weights.weights.h5')
        model.save_weights(final_weights_path)
        print(f"Final model weights saved to {final_weights_path}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback; traceback.print_exc()
else:
    print("No training data found. Skipping training.")

if history and history.history:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 1, 1)
    plt.plot(history.history['loss'], label='Loss')
    if val_dataset and 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss (MAE)'); plt.legend(); plt.title('Model Loss')
    plt.tight_layout(); plt.show()

if val_dataset:
    num_samples_to_show = min(2, GLOBAL_BATCH_SIZE if GLOBAL_BATCH_SIZE > 0 else 1)
    print("Generating predictions for visualization...")
    try:
        best_weights_loaded = False
        if hasattr(early_stopping_callback, 'best_weights') and early_stopping_callback.best_weights and early_stopping_callback.stopped_epoch > 0 :
             model.set_weights(early_stopping_callback.best_weights)
             print("Loaded best weights from EarlyStopping callback for prediction.")
             best_weights_loaded = True
        if not best_weights_loaded:
            eval_weights_path = os.path.join(CHECKPOINT_DIR, 'final_model_weights.weights.h5')
            if os.path.exists(eval_weights_path):
                 model.load_weights(eval_weights_path)
                 print(f"Loaded weights from final saved model for prediction: {eval_weights_path}")
            else:
                latest_ckpt_for_eval = tf.train.latest_checkpoint(CHECKPOINT_DIR)
                if latest_ckpt_for_eval:
                    try:
                        model.load_weights(latest_ckpt_for_eval)
                        print(f"Loaded weights from latest TF checkpoint for prediction: {latest_ckpt_for_eval}")
                    except Exception as e_load:
                        print(f"Could not load latest TF checkpoint {latest_ckpt_for_eval}: {e_load}. Using model's current weights.")
                else:
                    print("No specific best/final weights loaded for eval beyond current model state.")

        for lr_batch, hr_batch in val_dataset.take(1):
            if lr_batch.shape[0] == 0: print("Validation batch is empty."); break
            predictions_normalized = model.predict(lr_batch)
            for i in range(min(num_samples_to_show, lr_batch.shape[0])):
                lr_img_norm = lr_batch[i,...,0].numpy(); hr_img_norm = hr_batch[i,...,0].numpy(); pred_img_norm = predictions_normalized[i,...,0]
                plt.figure(figsize=(18,6)); plt.subplot(1,3,1); plt.imshow(lr_img_norm,cmap='viridis',vmin=0,vmax=1); plt.title("LR Input (Normalized)"); plt.colorbar(label="Norm. Value")
                plt.subplot(1,3,2); plt.imshow(pred_img_norm,cmap='viridis',vmin=0,vmax=1); plt.title("Model Prediction (Normalized)"); plt.colorbar(label="Norm. Value")
                plt.subplot(1,3,3); plt.imshow(hr_img_norm,cmap='viridis',vmin=0,vmax=1); plt.title("HR Ground Truth (Normalized)"); plt.colorbar(label="Norm. Value")
                plt.tight_layout(); plt.show()
            break
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback; traceback.print_exc()
else:
    print("No validation data to visualize.")
print("--- Full Script Execution Complete ---")
