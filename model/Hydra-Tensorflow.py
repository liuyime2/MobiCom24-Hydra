#!/usr/bin/python3
'''
This work is published in MobiCom 2024 Hydra: Accurate Multi-Modal Leaf Wetness Sensing with mm-Wave and Camera
This work is published under the MIT License.
'''

import tensorflow as tf
from tensorflow.keras.layers import (Input, Layer, Flatten, Concatenate, UpSampling2D, Conv2D, MaxPooling2D, 
                                     GlobalAveragePooling2D, GlobalAveragePooling1D, Dense, Activation, Dropout, 
                                     BatchNormalization, LayerNormalization, Add, MultiHeadAttention)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


# Multi-Modal Fusion Layer
class MaskScalingLayer(Layer):
    def __init__(self, **kwargs):
        super(MaskScalingLayer, self).__init__(**kwargs)
        self.dense = Dense(1, activation='linear')

    def call(self, inputs, **kwargs):
        mask, image = inputs
        flattened_mask = Flatten()(mask)
        scaling_factor = self.dense(flattened_mask)
        scaling_factor = tf.reshape(scaling_factor, [-1, 1, 1, 1])
        
        scaled_mask = mask * scaling_factor
        return image * scaled_mask  
    def compute_output_shape(self, input_shape):
        return input_shape[1]

# Residual Block for ResNet-18 Feature Extraction
def residual_block(x, filters, blocks, stride=1):
    shortcut = x
    for i in range(blocks):
        if i == 0:
            shortcut = Conv2D(filters, (1, 1), strides=(stride, stride))(shortcut)
            
        x = Conv2D(filters, (3, 3), strides=(stride, stride), padding='same')(x)
        x = Activation('relu')(x)
        stride = 1
    x = Add()([x, shortcut])
    return x

# ResNet-18 Feature Extractor
def resnet18_feature_extractor(input_feature):
    x = residual_block(input_feature, 64, 2)
    x = residual_block(x, 128, 2, stride=2)
    x = residual_block(x, 256, 2, stride=2) 
    x = residual_block(x, 512, 2, stride=2)
    return x

# Transformer Encoder Block
def transformer_encoder(inputs, num_heads, key_dim, ff_dim):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization()(attention_output)
    
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Add()([attention_output, ff_output])
    return LayerNormalization()(ff_output)

# Single Depth Feature Extraction for SAR and RGB fusion
def Single_Depth_FE(sar_input, image_input):
    mask_scaling_layer = MaskScalingLayer()
    fused_feature = mask_scaling_layer([sar_input, image_input])  # Fuse SAR and image data
    extracted_feature = resnet18_feature_extractor(fused_feature)  # Apply ResNet-like feature extractor
    extracted_feature = GlobalAveragePooling2D()(extracted_feature)  # Global pooling
    return extracted_feature
    
# Transformer Encoder Block
class TransformerEncoderBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ff_net = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, **kwargs):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=kwargs.get("training", False))
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection after attention
        ff_output = self.ff_net(out1)
        ff_output = self.dropout2(ff_output, training=kwargs.get("training", False))
        return self.layernorm2(out1 + ff_output)  # Final output with another residual connection

# Depth-aware Positional Encoding for Transformer
class DepthAwarePositionalEncoding(Layer):
    def __init__(self, depth, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.feature_dim = feature_dim
        self.pos_encoding = self.positional_encoding()  # Precompute positional encoding

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def positional_encoding(self):
        angle_rads = self.get_angles(tf.range(self.depth, dtype=tf.float32)[:, tf.newaxis],
                                     tf.range(self.feature_dim, dtype=tf.float32)[tf.newaxis, :],
                                     self.feature_dim)
        
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]  

        
# Main Hydra Model for Multi-Modal Leaf Wetness Sensing
def build_Hydra_model(sar_shape=(224, 224, 10), image_shape=(224, 224, 1), loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"], compile=True):
    sar_input = Input(shape=sar_shape, name='sar_input')
    image_input = Input(shape=image_shape, name='image_input')

    fused_features = [Single_Depth_FE(sar_input[:, :, :, i:i+1], image_input) for i in range(sar_shape[-1])]

    x = DepthAwarePositionalEncoding(depth=sar_shape[-1], feature_dim=512)(fused_features)
    
    transformer_block = TransformerEncoderBlock(embed_dim=512, num_heads=8, ff_dim=2048)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)  # Global pooling over the sequence

    output = Dense(1, activation='sigmoid')(x)  # Binary classification

    model = Model(inputs=[sar_input, image_input], outputs=output)

    if compile:
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

'''
Example setup of the Hydra model with SAR and image inputs
'''
if __name__ == "__main__":
    sar_shape = (224, 224, 10)
    image_shape = (224, 224, 1)
    
    model = build_Hydra_model(sar_shape, image_shape)
    model.summary()
