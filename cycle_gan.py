from lib.generator import Generator 
from lib.discriminator import Discriminator
from lib.utils import from_path_import
import tensorflow as tf
import torch
import os, sys
import data_pipeline
from official.nlp.transformer.misc import get_model_params
from types import SimpleNamespace
from official.nlp.transformer import embedding_layer
from logging import getLogger
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, GPT2LMHeadModel
import torch.nn.functional as F
from numba import cuda
from multiprocessing import Process, Queue

# from dictionary import Dictionary
from_path_import(
    name="dictionary",
    path="/home/zchen/encyclopedia-text-style-transfer/dictionary.py",
    globals=globals(),
    demands=["Dictionary"]
    )
    
sys.path.append("/home/zchen/encyclopedia-text-style-transfer/cnn-text-classification-pytorch/")
from model import CNN_Text
sys.path.pop()

logger = getLogger()

class cycle_gan:

    def __init__(self, args):

        # get model parameters
        params = get_model_params(args.param_set, 1)
        params.update(vars(args))
        params = SimpleNamespace(**params)

        # set vocabulary parameters
        vocab = Dictionary.read_vocab(args.vocab_file)
        self.vocab = params.vocab = vocab
        params.vocab_size = len(vocab)
        self.BOS_id = params.BOS_id = vocab.bos_index
        self.EOS_id = params.EOS_id = vocab.eos_index
        self.PAD_id = vocab.pad_index
        
        # misc
        params.max_io_parallelism = tf.data.experimental.AUTOTUNE
        params.repeat_dataset = None
        params.num_gpus = 1
        params.padded_decode = False
        params.dtype = tf.float32

        self.mode = params.mode
        self.out_dir = params.out_dir
        self.params = params
        
        # training configs
        self.pretrain_discriminator_steps = 0
        self.discriminator_iterations = params.dis_iter
        self.num_steps = params.num_steps
        self.ckpt_path = params.ckpt_path
        self.epoch_size = params.epoch_size
        self.log_interval = params.log_interval
        
        # Early stopping
        self.best_score = tf.Variable(0.)
        self.stopping_cnt = tf.Variable(0)
        self.early_stopping = params.early_stopping

        self.build_model()
        
    def build_model(self):
        # Build the generators.
        params = self.params
        hidden_size = params.hidden_size
        embedding = embedding_layer.EmbeddingSharedWeights(params.vocab_size, hidden_size)
        self.generator_X2Y, self.generator_Y2X = (
            Generator(params, embedding=embedding, name="generator_X2Y"),
            Generator(params, embedding=embedding, name="generator_Y2X")
            )
        self.models = [self.generator_X2Y, self.generator_Y2X]

        if self.mode == "train":
            # Build the discriminators.
            self.discriminator_X, self.discriminator_Y = (
                Discriminator(embedding=embedding, hidden_size=hidden_size, name="discriminator_X"),
                Discriminator(embedding=embedding, hidden_size=hidden_size, name="discriminator_Y")
            )
            self.models += [self.discriminator_X, self.discriminator_Y]

            self.load_eval_models()

        # Define the checkpoint content.
        ckpt_dict = {
            k: v
            for model in self.models
                for k, v in [
                    (model.name, model),
                    (f"{model.name}_opt", model.optimizer)
                    ]
        }
        checkpoint = tf.train.Checkpoint(**ckpt_dict, best_score=self.best_score, stopping_cnt=self.stopping_cnt)
            
        if self.ckpt_path is not None:
            # Load a checkpoint.
            logger.info("Loading the checkpoint ...")
            checkpoint.restore(self.ckpt_path).expect_partial()
        
        if self.mode == "train":
            # Load the last checkpoint.
            self.last_ckpt_manager = tf.train.CheckpointManager(
                checkpoint,
                directory=os.path.join(self.out_dir, "checkpoints/last/"),
                max_to_keep=1
                )
            self.best_ckpt_manager = tf.train.CheckpointManager(
                checkpoint,
                directory=os.path.join(self.out_dir, "checkpoints/best/"),
                max_to_keep=1
                )
            checkpoint.restore(self.last_ckpt_manager.latest_checkpoint)

    def load_eval_models(self):
        # style accuracy → CNN-based style classifier
        model = CNN_Text.from_pretrained("ETST", device="cpu")
        model.eval()
        self.style_classifier = model

        # content preservation → cos similarity(sentence transformer)
        model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1', device="cpu")
        model.eval()
        self.sentence_transformer = model

        # fluency → ppl(GPT2)
        self.gpt2_tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
        model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")    # default is cpu
        model.eval()
        self.gpt2 = model

        # # fluency → ppl(transformer decoder)
        # self.transformer_decoder = Generator(self.params, name="transformer_decoder").decode
        
    def train_step(self):
        # Train the discriminators.
        for i in range(self.discriminator_iterations):

            while True:
                try:
                    X, Y = next(self.data_iter)
                    batch_size = len(X)
                    loss_X, loss_Y = (
                        self.discriminator_X.step(self.generator_Y2X, Y, X, self.params),
                        self.discriminator_Y.step(self.generator_X2Y, X, Y, self.params)
                    )
                except tf.errors.ResourceExhaustedError:
                    logger.info("Out of GPU memory.")
                    logger.info("Try a new batch.")
                    continue
                break

            self.dis_X_loss += loss_X * batch_size
            self.dis_Y_loss += loss_Y * batch_size
            self.n_samples_d += batch_size
            logger.info(f"discriminator step {i + 1}")
            logger.info(f"discriminator_X loss: {loss_X:.4f}")
            logger.info(f"discriminator_Y loss: {loss_Y:.4f}")

        # Train the generators.
        if self.step >= self.pretrain_discriminator_steps:

            while True:
                try:
                    X, Y = next(self.data_iter)
                    batch_size = len(X)
                    loss_X2Y, loss_XYX, loss_Y2X, loss_YXY = Generator.step(
                        self.generator_X2Y, self.generator_Y2X, X, Y, self.discriminator_X, self.discriminator_Y, self.params
                        )
                except tf.errors.ResourceExhaustedError:
                    logger.info("Out of GPU memory.")
                    logger.info("Try a new batch.")
                    continue
                break
            
            self.xy_l += loss_X2Y * batch_size
            self.yx_l += loss_Y2X * batch_size
            self.xy_r += loss_XYX * batch_size
            self.yx_r += loss_YXY * batch_size
            self.n_samples_g += batch_size

            logger.info("generator step")
            logger.info(f"generator_X2Y loss: {loss_X2Y:.4f}")
            logger.info(f"XYX reconstruction loss: {loss_XYX:.4f}")
            logger.info("")
            logger.info(f"generator_Y2X loss: {loss_Y2X:.4f}")
            logger.info(f"YXY reconstruction loss: {loss_YXY:.4f}")
            logger.info("")
        
    def train(self):

        def better(a, b, mode):
            return a < b if mode == "min" else a > b

        log_interval = self.log_interval
        self.dis_X_loss = self.dis_Y_loss = self.xy_l = self.yx_l = self.xy_r = self.yx_r = self.n_samples_d = self.n_samples_g = 0
        training_set, valid_set = (
            data_pipeline.get_training_set(vars(self.params)),
            data_pipeline.get_valid_set(vars(self.params))
        )
        self.data_iter = iter(training_set)
        self.step = self.generator_X2Y.optimizer.iterations.numpy()

        # initial tensorboard file writers
        logdir = os.path.join(self.out_dir, "tensorboard/train")
        training_writer = tf.summary.create_file_writer(logdir)
        logdir = os.path.join(self.out_dir, "tensorboard/valid")
        valid_writer = tf.summary.create_file_writer(logdir)

        # start training
        while True:
            self.step += 1
            logger.info("")
            logger.info(f"training step: {self.step}")
            self.train_step()

            # make summary
            if self.step % log_interval == 0:
                # Save last.
                self.last_ckpt_manager.save(checkpoint_number=self.step)
                logger.info("Saved the last checkpoint.")
                logger.info("")

                step = self.step
                dis_X_loss = self.dis_X_loss / self.n_samples_d
                dis_Y_loss = self.dis_Y_loss / self.n_samples_d
                xy_l = self.xy_l / self.n_samples_g
                xy_r = self.xy_r / self.n_samples_g
                yx_l = self.yx_l / self.n_samples_g
                yx_r = self.yx_r / self.n_samples_g

                logger.info(f'step: {step}')
                logger.info(f'dis_X_loss: {dis_X_loss:.4f} dis_Y_loss: {dis_Y_loss:.4f}')
                logger.info(f'generator_X2Y_loss: {xy_l:.4f} XYX_reconstruction_loss {xy_r:.4f}')
                logger.info(f'generator_Y2X_loss {yx_l:.4f} YXY_reconstruction_loss {yx_r:.4f}')

                # write to tensorboard
                with training_writer.as_default():
                    tf.summary.scalar('discriminator X loss', data=dis_X_loss, step=step)
                    tf.summary.scalar('discriminator Y loss', data=dis_Y_loss, step=step)
                    tf.summary.scalar('generator X2Y loss', data=xy_l, step=step)
                    tf.summary.scalar('XYX reconstruction loss', data=xy_r, step=step)
                    tf.summary.scalar('generator Y2X loss', data=yx_l, step=step)
                    tf.summary.scalar('YXY reconstruction loss', data=yx_r, step=step)

                self.dis_X_loss = self.dis_Y_loss = self.xy_l = self.yx_l = self.xy_r = self.yx_r = self.n_samples_d = self.n_samples_g = 0
                
            if self.step % self.epoch_size == 0:
                with valid_writer.as_default():
                    self.evaluate(valid_set)

                    # Save best.
                    if better(self.valid_score, self.best_score, "max"):
                        self.best_score.assign(self.valid_score)
                        self.stopping_cnt.assign(0)
                        self.best_ckpt_manager.save(checkpoint_number=self.step)
                        logger.info(f"New best score: {self.best_score.numpy():.4f}")
                        logger.info("Saved the best checkpoint.")
                    else:
                        self.stopping_cnt.assign_add(1)
                        logger.info(f"The performance hasn't improved for {self.stopping_cnt.numpy()} / {self.early_stopping} epochs.")
                        logger.info(f"Best score: {self.best_score.numpy():.4f}")

                    # write to tensorboard
                    tf.summary.scalar('best valid score', data=self.best_score, step=self.step)
                    if self.stopping_cnt >= self.early_stopping:
                        logger.info("Early stopping ...")
                        exit()

            if self.num_steps is not None and self.step >= self.num_steps:
                break

    def tensor2txt(self, sent_batch):
        sentences = []
        for sent in sent_batch.numpy():
            txt = ""
            for id in sent:
                if id == self.BOS_id or id == self.PAD_id:
                    continue
                elif id == self.EOS_id:
                    break
                else:
                    txt += self.vocab[id]
            sentences.append(txt)
        return sentences

    @torch.no_grad()
    def evaluate(self, valid_set):
        logger.info("")
        logger.info("Start evaluation ...")
        step = self.step
        stats = {}
        for key in ["X2Y", "Y2X"]:
            stats[key] = {
                "style_accs": [],
                "content_preservation_scores": [],
                "fluency_scores": []
            }
            
        fake_X_batches, fake_Y_batches = [], []
        fake_X_txt, fake_Y_txt = [], []
        fake_X_file, fake_Y_file = (
            os.path.join(self.out_dir, f"generation/Y2X/fake_X-{step}.txt"),
            os.path.join(self.out_dir, f"generation/X2Y/fake_Y-{step}.txt")
        )

        # generate sentences
        if not hasattr(self, "X_txt"):
            """ generate real sentences """
            X_txt, Y_txt = [], []
            X_file, Y_file = (
                os.path.join(self.out_dir, "generation/X2Y/X.txt"),
                os.path.join(self.out_dir, "generation/Y2X/Y.txt")
            )

            for X, Y in valid_set:
                X_txt_batch = self.tensor2txt(X)
                Y_txt_batch = self.tensor2txt(Y)
                X_txt.append("".join([sent + '\n' for sent in X_txt_batch]))
                Y_txt.append("".join([sent + '\n' for sent in Y_txt_batch]))
                
            self.X_txt, self.Y_txt = X_txt, Y_txt
            if not os.path.exists(os.path.join(self.out_dir, "generation/")):
                os.makedirs(os.path.join(self.out_dir, "generation/X2Y/"))
                os.makedirs(os.path.join(self.out_dir, "generation/Y2X/"))

            with (
                open(X_file, 'w', encoding='utf-8') as fx,
                open(Y_file, 'w', encoding='utf-8') as fy
                ):
                fx.write("".join(X_txt))
                fy.write("".join(Y_txt))

        for (X, Y), X_txt, Y_txt in zip(valid_set, self.X_txt, self.Y_txt):
            fake_Y, fake_Y_txt_batch = self.eval_step(self.generator_X2Y, X)
            fake_X, fake_X_txt_batch = self.eval_step(self.generator_Y2X, Y)
            fake_X_batches.append(fake_X)
            fake_Y_batches.append(fake_Y)
            fake_X_txt.append("".join([sent + '\n' for sent in fake_X_txt_batch]))
            fake_Y_txt.append("".join([sent + '\n' for sent in fake_Y_txt_batch]))

            # write to tensorboard
            real = X_txt.strip().split('\n')[0]
            fake = fake_Y_txt_batch[0]
            tf.summary.text(f"(X2Y) {real}", fake, step=step)
            real = Y_txt.strip().split('\n')[0]
            fake = fake_X_txt_batch[0]
            tf.summary.text(f"(Y2X) {real}", fake, step=step)
            
        with (
            open(fake_X_file, 'w', encoding='utf-8') as fake_X_f,
            open(fake_Y_file, 'w', encoding='utf-8') as fake_Y_f
            ):
            fake_X_f.write("".join(fake_X_txt))
            fake_Y_f.write("".join(fake_Y_txt))

        # compute the metrics
        self.compute_metrics(fake_Y_batches, self.X_txt, fake_Y_txt, stats, "X2Y")
        self.compute_metrics(fake_X_batches, self.Y_txt, fake_X_txt, stats, "Y2X")
        
        for key, value in stats.items():
            logger.info('')
            logger.info(f"{key}:")
            sum_ = 0
            for k, v in value.items():
                avg = sum(v) / len(v)
                logger.info(f"{k}: {avg:.4f}")
                tf.summary.scalar(f"{key} {k}", data=avg, step=step)

                # Not using the fluency scores for validation.
                if k != "fluency_scores":
                    sum_ += avg

            value["avg"] = sum_ / 2
            logger.info(f"average: {value['avg']:.4f}")

        self.valid_score = (stats["X2Y"]["avg"] + stats["Y2X"]["avg"]) / 2
        logger.info('')
        logger.info(f"valid score: {self.valid_score:.4f}")
        tf.summary.scalar("valid score", data=self.valid_score, step=step)

    def eval_step(self, generator_X2Y, X):
        output = generator_X2Y([X], training=False, greedy_search=False)
        fake_Y = output["outputs"]
        fake_Y_txt = self.tensor2txt(fake_Y)
        return fake_Y, fake_Y_txt

    def compute_metrics(self, fake_Y_batches, X_txt_batches, fake_Y_txt_batches, stats, dir):
        stats = stats[dir]
        assert len(fake_Y_batches) == len(X_txt_batches) == len(fake_Y_txt_batches)
        for fake_Y, X_txt, fake_Y_txt in zip(fake_Y_batches, X_txt_batches, fake_Y_txt_batches):
            X_txt, fake_Y_txt = (
                X_txt.strip().split('\n'),
                fake_Y_txt.strip().split('\n')
            )
            
            # style accuracy → CNN-based style classifier
            fake_Y = torch.tensor(fake_Y.numpy())
            style = 0 if dir == "Y2X" else 1
            style_accs = self.compute_sa(fake_Y, style)
            stats["style_accs"] += style_accs

            # content preservation → cos similarity(sentence transformer)
            content_preservation_scores = self.compute_cp(X_txt, fake_Y_txt)
            stats["content_preservation_scores"] += content_preservation_scores

            # fluency → ppl(GPT2)
            fluency_scores = self.compute_fluency(fake_Y_txt)
            stats["fluency_scores"] += fluency_scores

    def compute_sa(self, sent_batch, style):
        output = self.style_classifier(sent_batch)   # (batch_size, 2)
        output = F.softmax(output, dim=1)
        return output[:, style].tolist()

    def compute_cp(self, real_batch, fake_batch):
        emb_real = self.sentence_transformer.encode(real_batch, show_progress_bar=False, convert_to_tensor=True)
        emb_fake = self.sentence_transformer.encode(fake_batch, show_progress_bar=False, convert_to_tensor=True)
        return F.cosine_similarity(emb_real, emb_fake).tolist()

    def compute_fluency(self, sent_batch):
        sent_batch = self.gpt2_tokenizer(
            sent_batch,
            padding=True,
            return_tensors="pt",
        )#.to("cuda")
        input_ids = sent_batch["input_ids"]
        attention_mask = sent_batch["attention_mask"]
        output = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = output["logits"]
        batch_size = len(logits)

        # Shift so that tokens < n predict n
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        attention_mask = attention_mask[:, 1:]

        # Flatten the tokens
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), input_ids.reshape(-1), reduction='none')    # (batch_size * seq_len,)
        loss = loss.reshape(batch_size, -1)    # (batch_size, seq_len)

        # compute the perplexity of each sentence
        length = attention_mask.sum(dim=1)
        loss *= attention_mask
        loss = loss.sum(dim=1) / length # (batch_size)
        return torch.exp(loss).tolist()

    # def compute_fluency(self, sent_batch):
    #     logits = self.transformer_decoder(targets=sent_batch, encoder_outputs=None, attention_bias=None, training=False)
    #     cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    #     loss = cross_entropy(sent_batch, logits)    # (batch_size, seq_len)

    #     # compute the perplexity of each sentence
    #     attention_mask = tf.cast(tf.not_equal(sent_batch, 0), tf.float32)
    #     length = tf.reduce_sum(attention_mask, axis=1)
    #     loss *= attention_mask
    #     loss = tf.reduce_sum(loss, axis=1) / length # (batch_size)
    #     print(loss)
    #     print(tf.shape(loss))
    #     return tf.exp(loss).numpy().tolist()

    def test(self):
        pass
    #     sentence = 'hi'
    #     gen_model_dir = os.path.join(self.model_dir,'generator/')
    #     self.sess.run(tf.global_variables_initializer())
    #     self.generator_saver.restore(self.sess, tf.train.latest_checkpoint(gen_model_dir))
    #     print('please enter one negative sentence')

    #     while(sentence):
    #         sentence = input('>')
    #         #sentence = sentence.split(':')[1]
    #         input_sent_vec = self.utils.sent2id(sentence)
    #         print(input_sent_vec)
    #         sent_vec = np.zeros((self.batch_size,self.sequence_length),dtype=np.int32)
    #         sent_vec[0] = input_sent_vec

    #         feed_dict = {
    #                 self.X2Y_inputs:sent_vec
    #         }
    #         preds = self.sess.run([self.X2Y_test_outputs],feed_dict)
    #         pred_sent = self.utils.vec2sent(preds[0][0])
    #         print(pred_sent)

    # def file_test(self):
    #     line_count = 0
    #     out_fp = open('test_out.txt','w')
    #     gen_model_dir = os.path.join(self.model_dir,'generator/')
    #     self.sess.run(tf.global_variables_initializer())
    #     self.generator_saver.restore(self.sess, tf.train.latest_checkpoint(gen_model_dir))
    #     for test_batch in self.utils.test_data_generator():
    #         feed_dict = {self.X2Y_inputs:test_batch}
    #         preds = self.sess.run([self.X2Y_test_outputs],feed_dict)
    #         preds = preds[0]
    #         for pred in preds:
    #             out_fp.write(self.utils.vec2sent(pred) + '\n')
    #             line_count += 1
    #             if line_count>=28658:
    #                 break
    #         if line_count>=28658:
    #             break