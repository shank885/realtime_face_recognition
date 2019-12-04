# triplet loss
import tensorflow as tf

def triplet_loss(y_true, y_pred, alpha=0.2):
    '''
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    '''
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # compute the encoding distance between the anchor and the positive,
    # need to sum over the axis -1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    # compute the encoding distance between the anchor and the negative
    # need to sum over the axis -1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    basic_loss = pos_dist - neg_dist + alpha
    # take the maximum of bsic loss and 0.0 sum over the training examples
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss


with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = ", str(loss.eval()))
