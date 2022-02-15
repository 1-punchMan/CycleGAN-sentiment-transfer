import tensorflow as tf
from official.nlp.transformer.transformer import Transformer
from official.nlp.transformer.metrics import transformer_loss
from logging import getLogger

logger = getLogger()

class Generator(Transformer):

    def __init__(self, params, embedding=None, name=None):
        super(Generator, self).__init__(vars(params), embedding=embedding, name=name)
        self.initialize_weights(params.transformer_path)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=params.learning_rate)

    def initialize_weights(self, transformer_path):
        inputs = tf.keras.layers.Input((None,), dtype="int64")
        targets = tf.keras.layers.Input((None,), dtype="int64")
        outputs = self([inputs, targets], training=True)
        model = tf.keras.Model([inputs, targets], outputs)

        if transformer_path is not None:
            # Load a pretrained transformer model.
            # For the saving formats that are not tf.train.Checkpoint and can be loaded by load_weights(), ex. keras ModelCheckpoint.
            logger.info(f"{self.name}: Loading the pretrained transformer model ...")
            load_status = model.load_weights(transformer_path)
            load_status.expect_partial()    # Silence the warnings.

            # `assert_consumed` can be used as validation that all variable values have been
            # restored from the checkpoint.
            # load_status.assert_consumed()

    @staticmethod
    def step(generator_X2Y, generator_Y2X, X, Y, discriminator_X, discriminator_Y, params):  

        def get_soft_seq(logits, b=10):
            """
            Convert logits to normalized probs.
            Use softmax to approximate one-hot.
            logits: (batch_size, seq_length, vocab_size)
            """
            
            return tf.nn.softmax(b * logits, axis=-1)

            # logits = tf.exp(b * logits)
            # return logits / tf.reduce_sum(logits, axis=-1, keepdims=True)  # (batch_size, seq_length, vocab_size)

        fake_Y = generator_X2Y([X], training=False, greedy_search=True)
        fake_X = generator_Y2X([Y], training=False, greedy_search=True)
        logger.info("fake_Y:")
        for sent in fake_Y.numpy():
            logger.info("".join([params.vocab[id] for id in sent]))
            break
        logger.info("fake_X:")
        for sent in fake_X.numpy():
            logger.info("".join([params.vocab[id] for id in sent]))
            break

        with tf.GradientTape(persistent=True) as tape:
            # XYX
            
            # Since the decoder will shift the target right and pad a zero on the left, the output length is the same as the target length.
            X2Y_outputs = generator_X2Y([X, fake_Y], training=True)
            X2Y_soft_outputs = get_soft_seq(X2Y_outputs)
            X2Y_soft_outputs = generator_Y2X.embedding_softmax_layer(X2Y_soft_outputs, mode="soft_seq")

            # reconstruction
            Y2X_rec_logits = generator_Y2X([fake_Y, X], enc_use_emb=False, enc_embedded_inputs=X2Y_soft_outputs, training=True)
            loss_XYX = transformer_loss(Y2X_rec_logits, X, params.label_smoothing, params.vocab_size)
            
            # YXY
            Y2X_outputs = generator_Y2X([Y, fake_X], training=True)
            Y2X_soft_outputs = get_soft_seq(Y2X_outputs)
            Y2X_soft_outputs = generator_X2Y.embedding_softmax_layer(Y2X_soft_outputs, mode="soft_seq")

            # reconstruction
            X2Y_rec_logits = generator_X2Y([fake_X, Y], enc_use_emb=False, enc_embedded_inputs=Y2X_soft_outputs, training=True)
            loss_YXY = transformer_loss(X2Y_rec_logits, Y, params.label_smoothing, params.vocab_size)
            
            false_Y_sample_score = discriminator_Y(X2Y_soft_outputs, training=False, use_emb=False)
            false_Y_sample_score = tf.reduce_mean(false_Y_sample_score)
            
            false_X_sample_score = discriminator_X(Y2X_soft_outputs, training=False, use_emb=False)
            false_X_sample_score = tf.reduce_mean(false_X_sample_score)

            reconstruction_loss = (loss_XYX + loss_YXY)*2.0
            loss_X2Y = reconstruction_loss - false_Y_sample_score
            loss_Y2X = reconstruction_loss - false_X_sample_score

        grads = tape.gradient(loss_X2Y, generator_X2Y.trainable_variables)
        generator_X2Y.optimizer.apply_gradients(zip(grads, generator_X2Y.trainable_variables))

        grads = tape.gradient(loss_Y2X, generator_Y2X.trainable_variables)
        generator_Y2X.optimizer.apply_gradients(zip(grads, generator_Y2X.trainable_variables))
        return loss_X2Y, loss_XYX, loss_Y2X, loss_YXY
