import tensorflow as tf


def roi_pooling_layer(feature_map, roi, output_shape, width=500., height=300.):

    def slice_by_roi(feature_map, roi, width, height):
        fw, fh = feature_map.get_shape().as_list()[1:3]
        # from roi to sliced (stretch usually < 1)
        w_stretch = tf.constant(fw / width, dtype=tf.float32)
        h_stretch = tf.constant(fh / height, dtype=tf.float32)

        left = tf.cast(roi[0] * w_stretch, dtype=tf.int32)
        right = tf.cast(roi[2] * w_stretch, dtype=tf.int32)
        down = tf.cast(roi[1] * h_stretch, dtype=tf.int32)
        up = tf.cast(roi[3] * h_stretch, dtype=tf.int32)

        sliced_feature_map = feature_map[:, left:right, down:up, :]
        return sliced_feature_map

    def sliced_to_output(sliced_feature_map, output_shape):
        sliced_shape = tf.shape(sliced_feature_map)
        sw = sliced_shape[1]
        sh = sliced_shape[2]

        # from output to slice point (stretch usually > 1, unless sliced shape < output_shape)
        w_stretch = sw / tf.constant(output_shape[0])
        h_stretch = sh / tf.constant(output_shape[1])

        w_slice_point = tf.range(output_shape[0] + 1, dtype=tf.float64) * w_stretch
        w_slice_point = tf.cast(w_slice_point, dtype=tf.int32)

        h_slice_point = tf.range(output_shape[1] + 1, dtype=tf.float64) * h_stretch
        h_slice_point = tf.cast(h_slice_point, dtype=tf.int32)

        output = []
        for w in range(output_shape[0]):
            output.append([])
            for h in range(output_shape[1]):
                output[-1].append(
                    tf.reduce_max(
                        sliced_feature_map[:,
                            w_slice_point[w]: w_slice_point[w + 1],
                            h_slice_point[h]: h_slice_point[h + 1], :
                        ],
                        axis=[1, 2]
                    )
                )
            output[-1] = tf.stack(output[-1], axis=1)
        output = tf.stack(output, axis=1)
        return output

    sliced_feature_map = slice_by_roi(feature_map, roi, width, height)
    output = sliced_to_output(sliced_feature_map, output_shape)
    return output

