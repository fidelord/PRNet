# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1  # Use TF-1.x graph/session features
import tf_slim as tcl              # tf_slim for conv2d/conv2d_transpose
import math
import cv2                         # For resizing

# Disable eager execution so we can use placeholders and sessions
tfv1.disable_eager_execution()

class PosPrediction:
    def __init__(self, resolution_inp=256, resolution_op=256):
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

        # Build network graph
        self.network_fn = resfcn256(self.resolution_inp)

        # Placeholder for input
        self.x = tfv1.placeholder(tf.float32, shape=[None, self.resolution_inp, self.resolution_inp, 3])
        self.x_op = self.network_fn(self.x, is_training=False)

        # Session setup
        gpu_options = tfv1.GPUOptions(allow_growth=True)
        config = tfv1.ConfigProto(gpu_options=gpu_options)
        self.sess = tfv1.Session(config=config)
        self.sess.run(tfv1.global_variables_initializer())

        # Attempt to restore pretrained weights
        prn_path = os.path.join('/content/PRNet/', 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.index'):
            print(f"Warning: Checkpoint not found at {prn_path}.index. Using initialized weights.")
        else:
            print(f"Restoring weights from: {prn_path}")
            saver = tfv1.train.Saver()
            try:
                for v in tfv1.global_variables():
                  print(v.name)
                saver.restore(self.sess, prn_path)
                print(f"Successfully restored weights from {prn_path}")
            except Exception as e:
                print(f"Error restoring weights: {e}. Using initialized weights.")

    def __call__(self, image_input_rgb):
        """
        Args:
            image_input_rgb (np.ndarray): RGB input [h, w, 3], values in [0..255]
        Returns:
            np.ndarray: Position map [h, w, 3], values in [-1..1]
        """
        h_orig, w_orig, _ = image_input_rgb.shape
        image_norm = image_input_rgb / 255.0 - 0.5

        # Resize if larger than input resolution
        if h_orig > self.resolution_inp or w_orig > self.resolution_inp:
            if h_orig > w_orig:
                new_h = self.resolution_inp
                new_w = int(w_orig * self.resolution_inp / h_orig)
            else:
                new_w = self.resolution_inp
                new_h = int(h_orig * self.resolution_inp / w_orig)
            image_resized = cv2.resize(image_norm, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image_norm

        pad_h = self.resolution_inp - image_resized.shape[0]
        pad_w = self.resolution_inp - image_resized.shape[1]
        image_padded = np.pad(
            image_resized,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant',
            constant_values=0.0
        )

        pos_map_padded = self.sess.run(self.x_op, feed_dict={self.x: image_padded[np.newaxis, :, :, :]})
        pos_map_padded = np.squeeze(pos_map_padded)
        pos_map_cropped = pos_map_padded[:image_resized.shape[0], :image_resized.shape[1], :]

        if h_orig > self.resolution_inp or w_orig > self.resolution_inp:
            pos_map_final = cv2.resize(pos_map_cropped, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        else:
            pos_map_final = pos_map_cropped

        return pos_map_final

    def close(self):
        if self.sess:
            self.sess.close()
            print("TensorFlow session closed.")


# -------------------------------------------------------------------------------
#                      Custom batch_norm_fn (using Keras BatchNormalization)
# -------------------------------------------------------------------------------


def batch_norm_fn(x, is_training, **kwargs):
    """
    Use tf.keras.layers.BatchNormalization under the current variable scope.
    This ensures that BN variables live under "<current_scope>/BatchNorm/*".
    """
    # Simply create the BatchNorm layer directly - no extra scope wrapping needed
    bn_layer = tf.keras.layers.BatchNormalization(
        momentum=0.999,
        epsilon=0.001,
        center=True,
        scale=True,
        trainable=True,
        name="BatchNorm"
    )
    return bn_layer(x, training=is_training)


# -------------------------------------------------------------------------------
#                                Networks
# -------------------------------------------------------------------------------
#                                Networks
# -------------------------------------------------------------------------------
def resfcn256(input_resolution=256):
    """
    Builds the PRNet-style 256×256 network with exactly ten encoder ResBlocks
    (each matching the GitHub channel/stride pattern) and five decoder ResBlocks,
    so that variable names line up with the original checkpoint.
    """
    def network_builder(x_input, is_training=True):
        print('Input tensor shape to network_builder:', x_input.get_shape())
        weights_reg = tf.keras.regularizers.l2(0.0002)

        with tfv1.variable_scope('resfcn256'):
            # ----------------------------------------------------------------
            #  Encoder: one initial Conv, then ten ResBlocks (down/sample pattern)
            # ----------------------------------------------------------------
            with tcl.arg_scope(
                [tcl.conv2d],
                activation_fn=tf.nn.relu,
                normalizer_fn=lambda x, **kw: batch_norm_fn(x, is_training, **kw),
                biases_initializer=None,
                padding='SAME',
                weights_regularizer=weights_reg
            ):
                # First 7×7→16 conv; scope="Conv"
                net = tcl.conv2d(
                    x_input,
                    num_outputs=16,
                    kernel_size=4,
                    stride=1,
                    scope='Conv'
                )

                # Ten‐block encoder (downsample→stay)
                net = resnet_block(net, name='resBlock',   num_outputs=32,  stride=2, is_training=is_training)  # 256→128, 16→32
                net = resnet_block(net, name='resBlock_1', num_outputs=32,  stride=1, is_training=is_training)  # 128→128, 32→32

                net = resnet_block(net, name='resBlock_2', num_outputs=64,  stride=2, is_training=is_training)  # 128→64, 32→64
                net = resnet_block(net, name='resBlock_3', num_outputs=64,  stride=1, is_training=is_training)  # 64→64, 64→64

                net = resnet_block(net, name='resBlock_4', num_outputs=128, stride=2, is_training=is_training)  # 64→32, 64→128
                net = resnet_block(net, name='resBlock_5', num_outputs=128, stride=1, is_training=is_training)  # 32→32, 128→128

                net = resnet_block(net, name='resBlock_6', num_outputs=256, stride=2, is_training=is_training)  # 32→16, 128→256
                net = resnet_block(net, name='resBlock_7', num_outputs=256, stride=1, is_training=is_training)  # 16→16, 256→256

                net = resnet_block(net, name='resBlock_8', num_outputs=512, stride=2, is_training=is_training)  # 16→8, 256→512
                net = resnet_block(net, name='resBlock_9', num_outputs=512, stride=1, is_training=is_training)  # 8→8,   512→512

            # ----------------------------------------------------------------
            #  Decoder: five ResBlocks (mirroring channel sizes) + conv2d_transpose
            # ----------------------------------------------------------------
            with tcl.arg_scope(
                [tcl.conv2d_transpose],
                activation_fn=tf.nn.relu,
                normalizer_fn=lambda x, **kwargs: batch_norm_fn(x, is_training, **kwargs),
                biases_initializer=None,
                padding='SAME',
                weights_regularizer=weights_reg
            ):
                # Input 'net' from encoder is 512 channels
                
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=512,  # CORRECTED (was 256) - Matches checkpoint [..., 512, 512]
                    kernel_size=4,
                    stride=1, 
                    scope='Conv2d_transpose'
                ) # Output: 512 channels

                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=256,  # CORRECTED (was 128) - Matches checkpoint [..., 256, 512]
                    kernel_size=4,
                    stride=2,
                    scope='Conv2d_transpose_1'
                ) # Output: 256 channels

                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=256,  # CORRECTED (was 64) - Matches checkpoint [..., 256, 256]
                    kernel_size=4,
                    stride=1,
                    scope='Conv2d_transpose_2'
                ) # Output: 256 channels

                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=256,  # CORRECTED (was 32) - Matches checkpoint [..., 256, 256]
                    kernel_size=4,
                    stride=1,
                    scope='Conv2d_transpose_3'
                ) # Output: 256 channels

                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=128,  # CORRECTED (was 16) - Matches checkpoint [..., 128, 256]
                    kernel_size=4,
                    stride=2,
                    scope='Conv2d_transpose_4'
                ) # Output: 128 channels

                # Final upsampling layers (these seemed to match the checkpoint already)
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=128, # Matches checkpoint [..., 128, 128]
                    kernel_size=4,
                    stride=1,
                    scope='Conv2d_transpose_5'
                ) # Output: 128 channels
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=128, # Matches checkpoint [..., 128, 128]
                    kernel_size=4,
                    stride=1,
                    scope='Conv2d_transpose_6'
                ) # Output: 128 channels
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=64,  # Matches checkpoint [..., 64, 128]
                    kernel_size=4,
                    stride=2,
                    scope='Conv2d_transpose_7'
                ) # Output: 64 channels
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=64,  # Matches checkpoint [..., 64, 64]
                    kernel_size=4,
                    stride=1,
                    scope='Conv2d_transpose_8'
                ) # Output: 64 channels
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=64,  # Matches checkpoint [..., 64, 64]
                    kernel_size=4,
                    stride=1,
                    scope='Conv2d_transpose_9'
                ) # Output: 64 channels
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=32,  # Matches checkpoint [..., 32, 64]
                    kernel_size=4,
                    stride=2,
                    scope='Conv2d_transpose_10'
                ) # Output: 32 channels
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=32,  # Matches checkpoint [..., 32, 32]
                    kernel_size=4,
                    stride=1,
                    scope='Conv2d_transpose_11'
                ) # Output: 32 channels
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=16,  # Matches checkpoint [..., 16, 32]
                    kernel_size=4,
                    stride=2,
                    scope='Conv2d_transpose_12'
                ) # Output: 16 channels
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=16,  # Matches checkpoint [..., 16, 16]
                    kernel_size=4,
                    stride=1,
                    scope='Conv2d_transpose_13'
                ) # Output: 16 channels
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=3,   # Matches checkpoint [..., 3, 16]
                    kernel_size=4,
                    stride=1,
                    scope='Conv2d_transpose_14'
                ) # Output: 3 channels
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=3,   # Matches checkpoint [..., 3, 3]
                    kernel_size=4,
                    stride=1,
                    scope='Conv2d_transpose_15'
                ) # Output: 3 channels
                net = tcl.conv2d_transpose(
                    net,
                    num_outputs=3,   # Matches checkpoint [..., 3, 3]
                    kernel_size=4,
                    stride=1,
                    scope='Conv2d_transpose_16'
                ) # Output: 3 channels

            output_tensor = tf.sigmoid(net, name='output_tanh')
            print('Output tensor shape from network_builder:', output_tensor.get_shape())

        return output_tensor

    return network_builder


# ----------------------------------------------------------------------------
#  Updated ResNet block that matches the GitHub “resBlock” semantics exactly.
# ----------------------------------------------------------------------------
def resnet_block(input_tensor, name, num_outputs, stride=1, is_training=True):
    """
    Builds a ResBlock using a 3-convolution main path (1x1, 4x4, 1x1)
    to match the apparent PRNet checkpoint structure.
    'num_outputs' is the final channel count for the block.
    The bottleneck channel depth is num_outputs // 2.
    """
    with tfv1.variable_scope(name):
        in_ch = input_tensor.get_shape().as_list()[-1]
        
        # Determine bottleneck channel depth
        # Based on checkpoint structure like:
        # resBlock (16_in -> 32_out): Conv(16->16), Conv1(16->16), Conv2(16->32) -> bottleneck = 16 = 32//2
        # resBlock_2 (32_in -> 64_out): Conv(32->32), Conv1(32->32), Conv2(32->64) -> bottleneck = 32 = 64//2
        bottleneck_depth = num_outputs // 2
        if bottleneck_depth == 0: # Handle cases like num_outputs=1 if it occurs, though unlikely for ResNet
            bottleneck_depth = num_outputs


        # --- Main Path ---
        # Layer 1: 1x1 conv, input_channels -> bottleneck_depth, stride 1
        current_main_path = tcl.conv2d(
            input_tensor,
            num_outputs=bottleneck_depth,
            kernel_size=1,
            stride=1,
            activation_fn=tf.nn.relu,
            normalizer_fn=lambda x, **kw_bn: batch_norm_fn(x, is_training, scope='Conv/BatchNorm', **kw_bn),
            biases_initializer=None,
            padding='SAME',
            weights_regularizer=tf.keras.regularizers.l2(0.0002),
            scope='Conv'
        )

        # Layer 2: 4x4 conv, bottleneck_depth -> bottleneck_depth, stride = block's stride
        current_main_path = tcl.conv2d(
            current_main_path,
            num_outputs=bottleneck_depth,
            kernel_size=4, # From checkpoint structure
            stride=stride, # Block's stride argument
            activation_fn=tf.nn.relu,
            normalizer_fn=lambda x, **kw_bn: batch_norm_fn(x, is_training, scope='Conv_1/BatchNorm', **kw_bn),
            biases_initializer=None,
            padding='SAME',
            weights_regularizer=tf.keras.regularizers.l2(0.0002),
            scope='Conv_1'
        )

        # Layer 3: 1x1 conv, bottleneck_depth -> num_outputs, stride 1
        # This is the final conv in the main path before addition, so no BN or ReLU here.
        main_path_net = tcl.conv2d(
            current_main_path,
            num_outputs=num_outputs, # Target output channels for the block
            kernel_size=1,
            stride=1,
            activation_fn=None, # No activation before residual addition
            normalizer_fn=None, # No batchnorm before residual addition
            biases_initializer=None,
            padding='SAME',
            weights_regularizer=tf.keras.regularizers.l2(0.0002),
            scope='Conv_2' # Corresponds to checkpoint variable name
        )
        
        # --- Shortcut Path ---
        shortcut_path = input_tensor
        # Project shortcut if input channels don't match target output channels OR if stride is > 1
        if in_ch != num_outputs or stride != 1:
            with tcl.arg_scope( # arg_scope for the shortcut convolution
                [tcl.conv2d],
                kernel_size=1,
                stride=stride, # Shortcut must also have the block's stride
                activation_fn=None,
                normalizer_fn=None, # Shortcut projection typically has no BN/ReLU
                padding='SAME',
                weights_regularizer=tf.keras.regularizers.l2(0.0002)
            ):
                shortcut_path = tcl.conv2d(input_tensor, num_outputs=num_outputs, scope='shortcut')
        
        # --- Fuse + final ReLU ---
        output_tensor = tf.nn.relu(main_path_net + shortcut_path, name='add_shortcut')
        
        return output_tensor
