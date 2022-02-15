"""
this is the GAN discriminator mudule
"""
import tensorflow as tf
from logging import getLogger

logger = getLogger()

class ResBlock(tf.keras.layers.Layer):
    
    def __init__(self, hidden_size):
        super(ResBlock, self).__init__()
        n_layers = 2
        self.layers_ = tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                    hidden_size, 3,
                    padding='SAME',
                    kernel_initializer='HeUniform',
                    activation='relu'
                ) for _ in range(n_layers)
            ]
        )

    def call(self, input):
        output = self.layers_(input)
        return input + 0.3*output

class Discriminator(tf.keras.Model):
    
    def __init__(self, embedding=None, hidden_size=None, name=None):
        super(Discriminator, self).__init__(name=name)
        self.embedding = embedding 
        n_res_blocks = 4
        self.layers_ = tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                    hidden_size, 1,
                    padding='SAME',
                    kernel_initializer='HeUniform',
                    activation='relu'
                )
                ] + [
                    ResBlock(hidden_size) for _ in range(n_res_blocks)
                ]
                ) 
        self.dropout = tf.keras.layers.Dropout(0.1) 
        self.linear = tf.keras.layers.Dense(1) 

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.initialize_weights()

    def initialize_weights(self):
        inputs = tf.keras.layers.Input((None,), dtype="int64")
        outputs = self(inputs, training=True, use_emb=True)
        tf.keras.Model(inputs, outputs)

    def call(self, input, training, use_emb=True, emb_mode="embedding"):
        """ {input} size: (batch_size, seq_len, hidden_size) """

        if use_emb:
            input = self.embedding(input, mode=emb_mode)
        tensor = self.layers_(input)
        tensor = tf.math.reduce_max(tensor, axis=1)    # (batch_size, hidden_size)
        tensor = self.linear(self.dropout(tensor, training=training))    # (batch_size, 1)

        return tf.squeeze(tensor,[1])    # (batch_size)

    def get_gradient_penalty(self, generator_outputs_embedded, real_sample_embedded):
        """ Set the gradient penalty. """

        batch_size = tf.shape(generator_outputs_embedded)[0]
        
        def pad(seq, length):
            pad_len = length - tf.shape(seq)[1]
            paddings = tf.zeros([batch_size, pad_len], dtype=tf.int32) # (batch_size, seq_len)
            paddings = self.embedding(paddings) # (batch_size, seq_len, hidden_size)

            return tf.concat([seq, paddings], axis=1) # (batch_size, seq_len, hidden_size)

        def pad_short_seq(seq1, seq2):
            len1, len2 = tf.shape(seq1)[1], tf.shape(seq2)[1]
            if len1 == len2:
                return seq1, seq2
            elif len1 > len2:
                return seq1, pad(seq2, len1)
            else:
                return seq2, pad(seq1, len2)

        generator_outputs_embedded, real_sample_embedded = pad_short_seq(generator_outputs_embedded, real_sample_embedded)
        alpha = tf.random.uniform(
            shape=[batch_size,1,1], 
            minval=0.,
            maxval=1.)
        differences = generator_outputs_embedded - real_sample_embedded
        interpolates = real_sample_embedded + (alpha*differences)

        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            output = self(interpolates, training=True, use_emb=False)

        gradients = tape.gradient(output, interpolates)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return gradient_penalty

    def step(self, generator, X, Y, params):   
        X2Y_outputs = generator([X], training=False, greedy_search=True)
        logger.info(f"{self.name}:")
        for sent in X2Y_outputs.numpy():
            logger.info("".join([params.vocab[id] for id in sent]))
            break
        with tf.GradientTape() as tape:
            X2Y_outputs, Y = (
                self.embedding(X2Y_outputs),
                self.embedding(Y)
            )
            false_Y_sample_score = self(X2Y_outputs, training=True, use_emb=False)
            real_Y_sample_score = self(Y, training=True, use_emb=False)
            dis_Y_penalty = self.get_gradient_penalty(X2Y_outputs, Y)
            loss = Discriminator.get_discriminator_loss(real_Y_sample_score,false_Y_sample_score,dis_Y_penalty)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    @staticmethod
    def get_discriminator_loss(real_sample_score,false_sample_score,gradient_penalty):
        real_sample_score = tf.reduce_mean(real_sample_score)
        false_sample_score = tf.reduce_mean(false_sample_score)
        discriminator_loss = -(real_sample_score - false_sample_score) + 10.0*gradient_penalty
        
        return discriminator_loss