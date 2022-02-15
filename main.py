import os
import argparse
from cycle_gan import cycle_gan
import tensorflow as tf
from lib.utils import from_path_import

# from logger import create_logger
name = "logger"
path = "/home/zchen/encyclopedia-text-style-transfer/logger.py"
demands = ["create_logger"]
from_path_import(name, path, globals(), demands)

def parse():
    #best: 0.5,0.01
    parser = argparse.ArgumentParser(description="cycle GAN")
    parser.add_argument('-out_dir','--out_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-n_tokens','--n_tokens',default=3000,type=int,help='# of tokens per batch')
    parser.add_argument('-wiki_dir','--wiki_dir',help='')
    parser.add_argument('-baidu_dir','--baidu_dir',help='')
    parser.add_argument('-vocab_file','--vocab_file',help='')
    parser.add_argument('-param_set', '--param_set', default="base", help='')
    parser.add_argument('-learning_rate', '--learning_rate', default=10e-4, type=float, help='transformer learning rate')
    parser.add_argument('-epoch_size','--epoch_size',default=1000,type=int,help='# of steps per epoch')
    parser.add_argument('-log_interval','--log_interval',default=100,type=int,help='log every # steps')
    parser.add_argument('-num_steps','--num_steps',type=int,help='number of steps')
    parser.add_argument('-n_valid_steps','--n_valid_steps',type=int,help='# of validation steps')
    parser.add_argument('-early_stopping','--early_stopping',type=int,help='early stopping')
    parser.add_argument('-max_length','--max_length',default=256,type=int,help='')
    parser.add_argument('-dis_iter','--dis_iter',default=3,type=int,help='discriminator iterations')
    parser.add_argument('-transformer_path','--transformer_path',help='Load a pretrained transformer model.')
    parser.add_argument('-ckpt_path','--ckpt_path',help='load a checkpoint')
    parser.add_argument('-mode', '--mode',default='train',help='train, file_test, test')
    parser.add_argument('-beam_size','--beam_size',default=1,type=int,help='')
    args = parser.parse_args()
    return args

def run(args):
    model = cycle_gan(args)
    if args.mode=='train':
        model.train()
    elif args.mode=='file_test':
        model.file_test()
    elif args.mode=='test':
       model.test()

if __name__ == '__main__':
    # tf.data.experimental.enable_debug_mode()
    args = parse()
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    create_logger(os.path.join(out_dir, "log"))
    run(args)
