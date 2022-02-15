import tensorflow as tf
from official.nlp.transformer.transformer import Transformer
from official.nlp.transformer.metrics import transformer_loss

class Generator(Transformer):

    def __init__(self, params, embedding=None, name=None):
        super(Generator, self).__init__(vars(params), embedding=embedding, name=name)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

    @staticmethod
    def step(generator_X2Y, generator_Y2X, X, Y, discriminator_X, discriminator_Y, params):   

        with tf.GradientTape(persistent=True) as tape:
            X2Y_outputs, X2Y_soft_outputs = generator_X2Y([X], training=True, greedy_search=True) # output (seq, soft_seq)
            X2Y_soft_outputs = generator_Y2X.embedding_softmax_layer(X2Y_soft_outputs, mode="soft_seq")
            Y2X_rec_logits = generator_Y2X([X2Y_outputs, X], enc_use_emb=False, enc_embedded_inputs=X2Y_soft_outputs, training=True)
            loss_XYX = transformer_loss(Y2X_rec_logits, X, params.label_smoothing, params.vocab_size)

            Y2X_outputs, Y2X_soft_outputs = generator_Y2X([Y], training=True, greedy_search=True) # output (seq, soft_seq)
            Y2X_soft_outputs = generator_X2Y.embedding_softmax_layer(Y2X_soft_outputs, mode="soft_seq")
            X2Y_rec_logits = generator_X2Y([Y2X_outputs, Y], enc_use_emb=False, enc_embedded_inputs=Y2X_soft_outputs, training=True)
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
