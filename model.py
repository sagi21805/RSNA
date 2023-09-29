import tensorflow as tf
from config import IMAGE_SIZE, BATCH_SIZE, L2_REGULATOR

with tf.device('/device:GPU:0'):

    def build_model():
        inputs = tf.keras.Input(shape=IMAGE_SIZE, batch_size=BATCH_SIZE)
        
        
        flat = tf.keras.layers.Flatten(input_shape = IMAGE_SIZE, batch_size=BATCH_SIZE)(inputs)
        #*bowel 

        bowel1 = tf.keras.layers.Dense(units = 64, activation = tf.keras.activations.selu, kernel_regularizer=L2_REGULATOR)(flat)
        bowel_drop = tf.keras.layers.Dropout(0.2)(bowel1)
        bowel_out = tf.keras.layers.Dense(name = "bowel", units = 1, activation = tf.keras.activations.sigmoid, kernel_regularizer=L2_REGULATOR)(bowel_drop)
        
        #*extravasation 

        extra1 = tf.keras.layers.Dense(units = 64, activation = tf.keras.activations.selu, kernel_regularizer=L2_REGULATOR)(flat)
        extra_drop = tf.keras.layers.Dropout(0.2)(extra1)
        extra_out = tf.keras.layers.Dense(name = "extra", units = 1, activation = tf.keras.activations.sigmoid, kernel_regularizer=L2_REGULATOR)(extra_drop)  
        
        #*kidney 

        kidney1 = tf.keras.layers.Dense(units = 64, activation = tf.keras.activations.selu, kernel_regularizer=L2_REGULATOR)(flat)
        kidney_drop = tf.keras.layers.Dropout(0.2)(kidney1)
        kidney_out = tf.keras.layers.Dense(name = "kidney", units = 3, activation = tf.keras.activations.softmax, kernel_regularizer=L2_REGULATOR)(kidney_drop)
        
        #*liver 

        liver1 = tf.keras.layers.Dense(units = 64, activation = tf.keras.activations.selu, kernel_regularizer=L2_REGULATOR)(flat)
        liver_drop = tf.keras.layers.Dropout(0.2)(liver1)
        liver_out = tf.keras.layers.Dense(name = "liver", units = 3, activation = tf.keras.activations.softmax, kernel_regularizer=L2_REGULATOR)(liver_drop)
        
        #*spleen 

        spleen1 = tf.keras.layers.Dense(units = 64, activation = tf.keras.activations.selu, kernel_regularizer=L2_REGULATOR)(flat)
        spleen_drop = tf.keras.layers.Dropout(0.2)(spleen1)
        spleen_out = tf.keras.layers.Dense(name = "spleen", units = 3, activation = tf.keras.activations.softmax, kernel_regularizer=L2_REGULATOR)(spleen_drop)

        outputs = [bowel_out, extra_out, kidney_out, liver_out, spleen_out]

        #*compile config
        optimizer = tf.keras.optimizers.Adam()

        loss = {
            "bowel":tf.keras.losses.BinaryCrossentropy(),
            "extra":tf.keras.losses.BinaryCrossentropy(),
            "liver":tf.keras.losses.CategoricalCrossentropy(),
            "kidney":tf.keras.losses.CategoricalCrossentropy(),
            "spleen":tf.keras.losses.CategoricalCrossentropy()
        }

        metrics = {
            "bowel":["accuracy"],
            "extra":["accuracy"],
            "liver":["accuracy"],
            "kidney":["accuracy"],
            "spleen":["accuracy"]
        }

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        print(model.summary())

        return model







    
    
