from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

import TensorflowUtils as utils
import read_in_data as scene_parsing
import datetime
import TFReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs3/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "/notebooks/FCN.tensorflow/dataroot/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_string('mainnet_dir', 'logs_mainnet/', 'If specified, mainnet will be restored from this directory')
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_bool('image_augmentation', "False", "Image augmentation: True/ False")
tf.flags.DEFINE_float('dropout', "0.5", "Probably of keeping value in dropout (valid values (0.0,1.0]")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
tf.flags.DEFINE_bool('tune_context', 'False', 'Tune context subnet')

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1.5e5)
NUM_OF_CLASSESS = 4
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def vgg_dilated(weights, image):

  layers = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
  )
  d_rate = [1, 1, 1, 1, 2]

  net = {}
  current = image
  for i, name in enumerate(layers):
    kind = name[:4]
    group = int(name[4])
    if kind == 'conv':
      if group in [1, 2, 3, 4]:
        kernels, bias = weights[i][0][0][0][0]
      else:
        kernels, bias = weights[i + 1][0][0][0][0]
      # matconvnet: weights are [width, height, in_channels, out_channels]
      # tensorflow: weights are [height, width, in_channels, out_channels]
      kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
      bias = utils.get_variable(bias.reshape(-1), name=name + "_b")

      current = utils.conv2d_dilated(current, kernels, bias, rate=d_rate[group-1])
        
    elif kind == 'relu':
      current = tf.nn.relu(current, name=name)
      if FLAGS.debug:
        utils.add_activation_summary(current)
    elif kind == 'pool':
      current = utils.avg_pool_2x2(current)
    net[name] = current

  return net

def context_net(weights, fmap):

  layers = (
    'dconv1', 'relu11', 'dconv2', 'relu12', 'dconv3', 'relu13',

    'dconv4', 'relu14', 'dconv5', 'relu15', 'dconv6', 'relu16',

    'dconv7', 'relu17', 'dconv8'
  )

  d_rate = [1, 1, 2, 4, 8, 16, 1, 1]

  cnet = {}
  current = fmap
  for i, name in enumerate(layers):
    kind = name[:-2]
    group = int(name[-1])
    if kind == 'conv':
      kernels, bias = weights[i]
      print(bias)
      # matconvnet: weights are [width, height, in_channels, out_channels]
      # tensorflow: weights are [height, width, in_channels, out_channels]
      kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
      bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
      current = utils.conv2d_dilated(current, kernels, bias, rate=d_rate[group - 1])

    elif kind == 'relu':
      current = tf.nn.relu(current, name=name)
      if FLAGS.debug:
        utils.add_activation_summary(current)

    cnet[name] = current
  return cnet

def inference(image, keep_prob, enable_context):
  """
  Semantic segmentation network definition
  :param image: input image. Should have values in range 0-255
  :param keep_prob:
  :return:
  """
  print("setting up vgg initialized conv layers ...")
  model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

  mean = model_data['normalization'][0][0][0]
  mean_pixel = np.mean(mean, axis=(0, 1))

  weights = np.squeeze(model_data['layers'])
  # Identity initialization
  weights_cnet = utils.get_cnet_data()

  processed_image = utils.process_image(image, mean_pixel)

  with tf.variable_scope("inference", reuse = tf.AUTO_REUSE):
    image_net = vgg_dilated(weights, processed_image)
    
    conv_final_layer = image_net['conv5_3']

    W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
    b6 = utils.bias_variable([4096], name="b6")
    conv6 = utils.conv2d_dilated(conv_final_layer, W6, b6, rate = 4)
    relu6 = tf.nn.relu(conv6, name="relu6")
    if FLAGS.debug:
      utils.add_activation_summary(relu6)
    relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

    W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
    b7 = utils.bias_variable([4096], name="b7")
    conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
    relu7 = tf.nn.relu(conv7, name="relu7")
    if FLAGS.debug:
      utils.add_activation_summary(relu7)
    relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

    W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
    b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
    conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
    
  # Contextual network part
  with tf.variable_scope("context", reuse = tf.AUTO_REUSE):
    Wd_1 = utils.weight_variable_cconv([3, 3, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="Wd_1")
    bd_1 = utils.bias_variable([NUM_OF_CLASSESS], name="bd_1")
    convd_1 = utils.conv2d_dilated(conv8, Wd_1, bd_1, rate=1)
    relud_1 = tf.nn.relu(convd_1, name='relud_1')
    
    Wd_2 = utils.weight_variable_cconv([3, 3, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="Wd_2")
    bd_2 = utils.bias_variable([NUM_OF_CLASSESS], name="bd_2")
    convd_2 = utils.conv2d_dilated(relud_1, Wd_2, bd_2, rate=1)
    relud_2 = tf.nn.relu(convd_2, name='relud_2')
    
    Wd_3 = utils.weight_variable_cconv([3, 3, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="Wd_3")
    bd_3 = utils.bias_variable([NUM_OF_CLASSESS], name="bd_3")
    convd_3 = utils.conv2d_dilated(relud_2, Wd_3, bd_3, rate=2)
    relud_3 = tf.nn.relu(convd_3, name='relud_3')
    
    Wd_4 = utils.weight_variable_cconv([3, 3, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="Wd_4")
    bd_4 = utils.bias_variable([NUM_OF_CLASSESS], name="bd_4")
    convd_4 = utils.conv2d_dilated(relud_3, Wd_4, bd_4, rate=4)
    relud_4 = tf.nn.relu(convd_4, name='relud_4')
    
#     Wd_5 = utils.weight_variable_cconv([3, 3, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="Wd_5")
#     bd_5 = utils.bias_variable([NUM_OF_CLASSESS], name="bd_5")
#     convd_5 = utils.conv2d_dilated(relud_4, Wd_5, bd_5, rate=8)
#     relud_5 = tf.nn.relu(convd_5, name='relud_5')
    
#     Wd_6 = utils.weight_variable_cconv([3, 3, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="Wd_6")
#     bd_6 = utils.bias_variable([NUM_OF_CLASSESS], name="bd_6")
#     convd_6 = utils.conv2d_dilated(relud_5, Wd_6, bd_6, rate=16)
#     relud_6 = tf.nn.relu(convd_6, name='relud_6')
    
    Wd_7 = utils.weight_variable_cconv([3, 3, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="Wd_7")
    bd_7 = utils.bias_variable([NUM_OF_CLASSESS], name="bd_7")
    convd_7 = utils.conv2d_dilated(relud_4, Wd_7, bd_7, rate=1)
    relud_7 = tf.nn.relu(convd_7, name='relud_7')
    
    Wd_8 = utils.weight_variable_cconv([1, 1, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="Wd_8")
    bd_8 = utils.bias_variable([NUM_OF_CLASSESS], name="bd_8")
    convd_8 = utils.conv2d_dilated(relud_7, Wd_8, bd_8, rate=1)

  with tf.variable_scope("final", reuse = tf.AUTO_REUSE):
    # now to upscale to actual image size
    shape = tf.shape(image)
    deconv_shape = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
    W_t = utils.weight_variable([16, 16, NUM_OF_CLASSESS, NUM_OF_CLASSESS], name="W_t")
    b_t = utils.bias_variable([NUM_OF_CLASSESS], name="b_t")
    if enable_context == False:
        conv_t3 = utils.conv2d_transpose_strided(conv8, W_t, b_t, output_shape=deconv_shape, stride=8)
    else:
        conv_t3 = utils.conv2d_transpose_strided(convd_8, W_t, b_t, output_shape=deconv_shape, stride=8)

    annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

  return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list, lr = FLAGS.learning_rate):
  optimizer = tf.train.AdamOptimizer(lr)
  grads = optimizer.compute_gradients(loss_val, var_list=var_list)
  if FLAGS.debug:
    for grad, var in grads:
      utils.add_gradient_summary(grad, var)
  return optimizer.apply_gradients(grads)


def main(argv=None):
    with tf.device('/device:GPU:0'):
        keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_image")
        annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

        pred_annotation, logits = inference(image, keep_probability, enable_context = False)
        pred_annotation_ctx, logits_ctx = inference(image, keep_probability, enable_context = True)
        tf.summary.image("input_image", image, max_outputs=FLAGS.batch_size)
        tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=FLAGS.batch_size)
        tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=FLAGS.batch_size)
        tf.summary.image("pred_annotation_ctx", tf.cast(pred_annotation_ctx, tf.uint8), max_outputs=FLAGS.batch_size)
        loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                              name="entropy")))
        loss_ctx = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_ctx,
                                                                              labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                              name="entropy_ctx")))
        
        loss_summary = tf.summary.scalar("entropy", loss)
        loss_summary_ctx = tf.summary.scalar("entropy", loss_ctx)
        
        # Now all training will be done in one run
        if FLAGS.tune_context == True:
            trainable_var_ctx = tf.trainable_variables(scope = 'context') + tf.trainable_variables(scope = 'final')
            ctx_train_op = train(loss_ctx, trainable_var_ctx, lr = FLAGS.learning_rate*10)
            ctx_train_op_ft = train(loss_ctx, trainable_var_ctx, lr = FLAGS.learning_rate)
            print('Context network enabled')
        trainable_var = tf.trainable_variables(scope = 'inference') + tf.trainable_variables(scope = 'final')
        train_op = train(loss, trainable_var, lr = FLAGS.learning_rate)
        
        print(trainable_var)
        if FLAGS.debug:
            for var in trainable_var:
                utils.add_to_regularization_and_summary(var)

        print("Setting up summary op...")
        summary_op = tf.summary.merge_all()

        print("Setting up image reader...")
    
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print("No. train records: ", len(train_records))
    print("No. validation records: ", len(valid_records))

    print("Setting up dataset reader")
    image_options_train = {'resize': True, 'resize_width': IMAGE_WIDTH, 'resize_height': IMAGE_HEIGHT, 'image_augmentation':FLAGS.image_augmentation}
    image_options_val = {'resize': True, 'resize_width': IMAGE_WIDTH, 'resize_height': IMAGE_HEIGHT}
    if FLAGS.mode == 'train':
        train_val_dataset = dataset.TrainVal.from_records(
            train_records, valid_records, image_options_train, image_options_val, FLAGS.batch_size, FLAGS.batch_size)

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config = config)

        print("Setting up Saver...")
            
        saver = tf.train.Saver()

        # create two summary writers to show training loss and validation loss in the same graph
        # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
        train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        trained_mainnet = False
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        else:
            ckpt_main = tf.train.get_checkpoint_state(FLAGS.mainnet_dir)
            if ckpt_main and ckpt_main.model_checkpoint_path:
                saver_main = tf.train.Saver(var_list = tf.trainable_variables(scope = 'inference'))
                saver_main.restore(sess, ckpt_main.model_checkpoint_path)
                trained_mainnet = True
                print("MainNet model restored...")
        
        if FLAGS.mode == "train":
            it_train, it_val = train_val_dataset.get_iterators()
            if FLAGS.dropout <=0 or FLAGS.dropout > 1:
                raise ValueError("Dropout value not in range (0,1]")

            #Ignore filename from reader
            next_train_images, next_train_annotations, next_train_name = it_train.get_next()
            next_val_images, next_val_annotations, next_val_name = it_val.get_next()
            
            if not trained_mainnet:
                for i in xrange(int(MAX_ITERATION // 3 + 1)):

                    train_images, train_annotations = sess.run([next_train_images, next_train_annotations])
                    feed_dict = {image: train_images, annotation: train_annotations, keep_probability: (1 - FLAGS.dropout)}

                    sess.run(train_op, feed_dict=feed_dict)

                    if i % 10 == 0:
                        train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                        print("MainNet -- Step: %d, Train_loss:%g" % (i, train_loss))
                        train_writer.add_summary(summary_str, i)

                    if i % 500 == 0:

                        valid_images, valid_annotations = sess.run([next_val_images, next_val_annotations])
                        valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                                                               keep_probability: 1.0})
                        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                        # add validation loss to TensorBoard
                        validation_writer.add_summary(summary_sva, i)
                        saver.save(sess, FLAGS.logs_dir + "model.ckpt", i)
            if FLAGS.tune_context:        
                for i in xrange(int(MAX_ITERATION // 3 + 1)):

                    train_images, train_annotations = sess.run([next_train_images, next_train_annotations])
                    feed_dict = {image: train_images, annotation: train_annotations, keep_probability: (1 - FLAGS.dropout)}

                    sess.run(ctx_train_op, feed_dict=feed_dict)

                    if i % 10 == 0:
                        train_loss, summary_str = sess.run([loss_ctx, loss_summary_ctx], feed_dict=feed_dict)
                        print("ContExt -- Step: %d, Train_loss:%g" % (i, train_loss))
                        train_writer.add_summary(summary_str, i)

                    if i % 500 == 0:

                        valid_images, valid_annotations = sess.run([next_val_images, next_val_annotations])
                        valid_loss, summary_sva = sess.run([loss_ctx, loss_summary_ctx], feed_dict={image: valid_images, annotation: valid_annotations,
                                                               keep_probability: 1.0})
                        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                        # add validation loss to TensorBoard
                        validation_writer.add_summary(summary_sva, i)
                        saver.save(sess, FLAGS.logs_dir + "model_context.ckpt", i)

                for i in xrange(int(MAX_ITERATION // 3 + 1)):

                    train_images, train_annotations = sess.run([next_train_images, next_train_annotations])
                    feed_dict = {image: train_images, annotation: train_annotations, keep_probability: (1 - FLAGS.dropout)}

                    sess.run(ctx_train_op_ft, feed_dict=feed_dict)

                    if i % 10 == 0:
                        train_loss, summary_str = sess.run([loss_ctx, loss_summary_ctx], feed_dict=feed_dict)
                        print("ContExt -- Step: %d, Train_loss:%g" % (i+int(MAX_ITERATION // 3), train_loss))
                        train_writer.add_summary(summary_str, i)

                    if i % 500 == 0:

                        valid_images, valid_annotations = sess.run([next_val_images, next_val_annotations])
                        valid_loss, summary_sva = sess.run([loss_ctx, loss_summary_ctx], feed_dict={image: valid_images, annotation: valid_annotations,
                                                               keep_probability: 1.0})
                        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                        # add validation loss to TensorBoard
                        validation_writer.add_summary(summary_sva, i)
                        saver.save(sess, FLAGS.logs_dir + "model_context.ckpt", i)


        elif FLAGS.mode == "visualize":
            iterator = train_val_dataset.get_iterator()
            get_next = iterator.get_next()
            training_init_op, val_init_op = train_val_dataset.get_ops()
            sess.run(val_init_op)
            valid_images, valid_annotations = sess.run(get_next)
            pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                        keep_probability: 1.0})
            valid_annotations = np.squeeze(valid_annotations, axis=3)
            pred = np.squeeze(pred, axis=3)

            for itr in range(FLAGS.batch_size):
                utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
                utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
                utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
                print("Saved image: %d" % itr)

        elif FLAGS.mode == "predict":
            predict_records = scene_parsing.read_prediction_set(FLAGS.data_dir)
            no_predict_images = len(predict_records)
            print ("No. of predict records {}".format(no_predict_images))
            predict_image_options = {'resize': True, 'resize_width': IMAGE_WIDTH, 'resize_height': IMAGE_HEIGHT, 'predict_dataset': True}
            test_dataset_reader = dataset.SingleDataset.from_records(predict_records, predict_image_options)
            next_test_image = test_dataset_reader.get_iterator().get_next()
            if not os.path.exists(os.path.join(FLAGS.logs_dir, "predictions")):
                os.makedirs(os.path.join(FLAGS.logs_dir, "predictions"))
#                 os.makedirs(os.path.join(FLAGS.logs_dir, "predictions_prob"))
            for i in range(no_predict_images):
                if (i % 10 == 0):
                    print("Predicted {}/{} images".format(i, no_predict_images))
                predict_images, predict_names = sess.run(next_test_image)
                pred, logits = sess.run([pred_annotation_ctx, logits_ctx], feed_dict={image: predict_images,
                                                            keep_probability: 1.0})
                logits = logits[0, ...]
                pred = np.squeeze(pred, axis=3)
#                 print(np.unique(pred))
                utils.save_image((pred[0] * (255 / 3)).astype(np.uint8), os.path.join(FLAGS.logs_dir, "predictions"),
                                 name="predict_" + str(predict_names))
#                 np.save(os.path.join(FLAGS.logs_dir, "predictions", "predict_prob_" + str(predict_names)), logits)


if __name__ == "__main__":
  tf.app.run()
