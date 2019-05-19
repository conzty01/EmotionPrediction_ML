'''
    --- Emotion Prediction with a Long-Short Term Memory RNN Model ---

    Guide for reading this program:
        """ ... """ : Used to explain general ideas or procedures that
                        encompass multiple blocks of code

        # ...       : Used to explain the following block of code

    Function Breakdown:
        getData     : Import and clean up the data from a .csv file
        createModel : Create, train, and test the LSTM RNN model
        main        : Call createModel
'''

import numpy as np
import pandas as pd
import tensorflow as tf

# Get rid of warning messages about soon-to-be depreciated functions
tf.logging.set_verbosity(tf.logging.ERROR)

def getData():
    """Import and clean up our data"""

    importedData = pd.read_csv("alldata.csv")

    # Remove all the random NaN that show up from importing
    cleanedData = importedData[np.isfinite(importedData["meanHrECG"])]
    cleanedData = cleanedData.drop(columns=['Unnamed: 0', 'subject'])

    # Split the data into train and test
    test_data = cleanedData.sample(frac=0.2)
    test_labels = test_data[['stress', 'amuse']]

    train_data = cleanedData.drop(test_data.index)
    train_labels = train_data[['stress', 'amuse']]

    # Remove the labels from the data
    test_data.drop(test_data[["stress", "amuse"]], axis=1, inplace=True)
    train_data.drop(train_data[["stress", "amuse"]], axis=1, inplace=True)

    # Organize the data into batches of 1000
    ret = []
    for d in [train_data.values, test_data.values, train_labels.values, test_labels.values]:
        res = []
        st = 0
        end = 1000

        while end <= d.size:
            res.append(d[st:end])

            st += 1000
            end += 1000

        ret.append(np.array(res))

    return ret


def createModel():
    """Create the machine learning model and train and test it
       (This might as well have been called main for its function)"""

    # Get the data from our source and split it into train and test data
    data = getData()
    train_data = data[0]
    test_data = data[1]
    train_labels = data[2]
    test_labels = data[3]

    sess = tf.Session()
    """
    Set up the LSTM model

    It is a best practice to create placeholders before variable assignments when using TensorFlow.
    Here we'll create placeholders for inputs ("Xs") and outputs ("Ys").

    Placeholder 'X': represents the "space" allocated input.

    Each input (row of our csv file) has 74 attributes that act as the input.
    The 'shape' argument defines the tensor size by its dimensions.
    1st dimension = 1000. Indicates that the batch size, is 1000 items long.
    2nd dimension = 74. Indicates the number of attributes in a single row.
    Placeholder 'Y': represents the final output or the labels.

    2 possible classes (1, 2)
    The 'shape' argument defines the tensor size by its dimensions.
    1st dimension = None. Indicates that the batch size, can be of any size.
    2nd dimension = 2. Indicates the number of targets/outcomes
    dtype for both placeholders: if you not sure, use tf.float32. The limitation here is that the later
    presented softmax function only accepts float32 or float64 dtypes.
    For more dtypes, check TensorFlow's documentation here
    """

    """Set up various variables for the project"""
    max_grad_norm = 5

    # The total number of recurrence steps, also known as the number of layers when our RNN is "unfolded"
    num_steps = 74
    # The number of processing units (neurons) in the hidden layers
    hidden_size_l1 = 256
    hidden_size_l2 = 128

    # The size for each batch of data
    batch_size = 1000
    # The size of our vocabulary
    vocab_size = 2

    # Create the x and y variables using our data
    x = train_data[0]
    y = train_labels[0]
    reshape_value = [1000,1,74]
    x = x.reshape(reshape_value)
    #y = y.reshape(reshape_value)

    _input_data = tf.placeholder(tf.int64, [batch_size, 1, num_steps])
    _targets = tf.placeholder(tf.int64, [batch_size, 2])

    # Assemble x and y into a feed_dict
    feed_dict = {_input_data: x, _targets: y}

    #print(sess.run(_input_data, feed_dict))

    # In this step, we create the stacked LSTM, which is a 2 layer LSTM network:
    lstm_cell_l1 = tf.contrib.rnn.BasicLSTMCell(hidden_size_l1, forget_bias=0.0)
    lstm_cell_l2 = tf.contrib.rnn.BasicLSTMCell(hidden_size_l2, forget_bias=0.0)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell_l1, lstm_cell_l2])

    _initial_state = stacked_lstm.zero_state(batch_size, tf.float64)

    #print(sess.run(_initial_state, feed_dict))

    # Create the RNN
    outputs, new_state = tf.nn.dynamic_rnn(stacked_lstm, x, initial_state=_initial_state)
    print(outputs)

    # Test run with a single data row just to make sure things are working
    sess.run(tf.global_variables_initializer())
    sess.run(outputs[0], feed_dict)

    """Softmax is used to take the outputs from the model and normalizes them
       to create a probability vector depicting whether or not a data row is 
       amused or stressed."""

    output = tf.reshape(outputs, [-1, hidden_size_l2])

    softmax_w = tf.get_variable("softmax_w", [hidden_size_l2, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(tf.cast(output, tf.float32), softmax_w) + softmax_b

    # Apply the softmax function
    prob = tf.nn.softmax(logits)
    sess.run(tf.global_variables_initializer())
    output_emotion_prob = sess.run(prob, feed_dict)
    print("shape of the output: ", output_emotion_prob.shape)
    print("The probability of observing words in t=0 to t=20", output_emotion_prob[0:20])

    # Run our model with actual test data to make sure things are going well
    targ = sess.run(_targets, feed_dict)
    print(targ[0])

    print(tf.reshape(_targets, [-1]).shape)
    print(logits.shape)
    print(_targets)

    # Calculate the loss of our data and model
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.slice(tf.reshape(_targets, [-1]), [0], [1000])],
                                                              [tf.ones([batch_size])])

    sess.run(loss, feed_dict)

    cost = tf.reduce_sum(loss) / batch_size
    sess.run(tf.global_variables_initializer())
    print(sess.run(cost, feed_dict))

    lr = tf.Variable(.05, trainable=False)

    """ The following lines create a gradient descent optimizer to 
        actually train our model to minimize the cost of our model"""
    optimizer = tf.train.GradientDescentOptimizer(lr)

    tvars = tf.trainable_variables()

    var_x = tf.placeholder(tf.float64)
    var_y = tf.placeholder(tf.float64)
    func_test = 2.0 * var_x * var_x + 3.0 * var_x * var_y
    sess.run(tf.global_variables_initializer())
    sess.run(func_test, {var_x: 1.0, var_y: 2.0})

    var_grad = tf.gradients(func_test, [var_x])
    sess.run(var_grad, {var_x: 1.0, var_y: 2.0})

    """Find the gradient models that work best for this model"""
    grad_t_list = tf.gradients(cost, tvars)

    grad_t_list = [tf.cast(t, tf.float64) for t in grad_t_list]

    grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)
    sess.run(grads, feed_dict)

    # Finangle the types so that all types are the same -> tf.float64
    tvars = [tf.cast(t, tf.float64) for t in tvars]

    # Run the optimizer to train our model
    z = zip(grads, tvars)
    train_op = optimizer.apply_gradients(z)
    sess.run(tf.global_variables_initializer())
    sess.run(train_op, feed_dict)

    # Calculate the accuracy of the model
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(_targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    """ Test our model using our test data """
    for i in range(20):
        for step in range(len(train_data[1:49])):
            x1 = train_data[step]
            y1 = train_labels[step]
            print(step)

            reshape_value = [1000, 1, 74]
            x1 = x1.reshape(reshape_value)
            a = sess.run(train_op, feed_dict={_input_data: x1, _targets: y1})
            print(sess.run(accuracy, feed_dict={_input_data: x1, _targets: y1}))


    for step in range(len(test_data[:12])):
        x1 = test_data[step]
        y1 = test_labels[step]
        print(step, len(test_data))

        reshape_value = [1000, 1, 74]
        x1 = x1.reshape(reshape_value)
        print(sess.run(accuracy, feed_dict={_input_data: x1, _targets: y1}))


def main():
    createModel()


if __name__ == "__main__":
    main()