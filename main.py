import os.path
import shutil
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

print()
print("Running test cases: 4 -->")

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    #  Implement function
    #  Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    graph = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    vgg_input_image_tensor = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_image_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output: shape=(?, ?, ?, 256) = (batch_size, w, h, depth)
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output: shape=(?, ?, ?, 512)
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output: shape=(?, ?, ?, 4096)
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    #Apply 1x1 convolution in place of fully connected layer
    #input=[5, 18, 4096]
    #out=[5, 18, 2]
    fcn8 = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=1, name="fcn8")

    #Upsample fcn8 with size depth=(4096?) to match size of layer 4
    #so that we can add skip connection with 4th layer
    #input=[5, 18, 2]
    #out=[10, 36, 512]
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=vgg_layer4_out.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    #add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, vgg_layer4_out, name="fcn9_plus_vgg_layer4")

    #upsample again
    #input=[10, 36, 512]
    #out=[20, 72, 256]
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=vgg_layer3_out.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    #add skip connection
    fcn10_skipp_connected = tf.add(fcn10, vgg_layer3_out, name="fcn10_plus_vgg_layer3")

    #upsample again
    #input=[20, 72, 256]
    #out=[160, 576, 2]
    fcn11 = tf.layers.conv2d_transpose(fcn10_skipp_connected, filters=num_classes,
    kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

    return fcn11

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    #Find logits --> reshape last layer so that rows represents all pixels and
    #columns represents classes
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

    #calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    #take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    #optimizer to reduce loss
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")
    return logits, train_op, loss_op

tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    keep_prob_value = 0.5
    learning_rate_value = 0.001
    for epoch in range(epochs):
        #batches, gt_batches = get_batches_fn(batch_size)
        # Create function to get batches
        # get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        total_loss = 0
        for X_batch, gt_batch in get_batches_fn(batch_size):

            loss, _ = sess.run([cross_entropy_loss, train_op],
            feed_dict={input_image: X_batch, correct_label: gt_batch,
            keep_prob: keep_prob_value, learning_rate:learning_rate_value})

            total_loss += loss;

        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))
        print()

#test case for method train_nn()
tests.test_train_nn(train_nn)


def save_model(sess):
    save_file_path = 'saved_model/model.ckpt'
    # Clean saved_model dir
    if os.path.exists(save_file_path):
        shutil.rmtree(save_file_path)
    os.makedirs(save_file_path)

    #saver to save the trained model
    saver = tf.train.Saver()
    #save model
    saver.save(sess, save_file_path)

def run():
    EPOCHS = 20
    BATCH_SIZE = 16
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    tests.test_for_kitti_dataset(data_dir)

    print("All test cases passed. Starting building FCN")

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)


    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    correct_label = tf.placeholder(tf.float32, name="correct_label")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        fcn = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        #build an optimizer
        logits, train_op, cross_entropy_loss_op = optimize(fcn, correct_label, learning_rate, num_classes)

        #initialize variables of FCN layers we just created
        sess.run(tf.global_variables_initializer())

        print("Model build successful, starting training")
        # Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss_op,
        input_image, correct_label, keep_prob, learning_rate)

        #let's save model
        print('Training successfull, saving model...')
        save_model(sess)
        print("Model saved.")

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        print("All done!")

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
