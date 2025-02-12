import tensorflow.compat.v1 as tf

def discriminator_lstm(inputs,lstm_length,vocab_size):
    with tf.variable_scope("discriminator_word_embedding") as scope:
        init = tf.keras.initializers.GlorotNormal()
        discriminator_word_embedding_matrix = tf.get_variable(
            name="word_embedding_matrix",
            shape=[vocab_size,300],
            initializer=init,
            trainable = True
        )
        inputs = tf.nn.embedding_lookup(discriminator_word_embedding_matrix,inputs)
    
    cell = tf.nn.rnn_cell.LSTMCell(num_units=500, state_is_tuple=True)
    lstm_outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float32,
        sequence_length=lstm_length,
        inputs=inputs)
    with tf.variable_scope("output_project") as scope:
        outputs = tf.layers.dense(lstm_outputs, 1, scope=scope)
    return tf.squeeze(outputs,axis=2)