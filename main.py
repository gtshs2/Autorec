from data_preprocessor import *
from AutoRec import AutoRec
import tensorflow as tf
import time
import argparse
current_time = time.time()

parser = argparse.ArgumentParser(description='custom AutoRec ')
parser.add_argument('--train_epoch', type=int, default=20)
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--lambda_value', type=float, default=1)
parser.add_argument('--random_seed', type=int, default=100)
parser.add_argument('--optimizer_method', choices=['Adam','Adadelta','Adagrad','RMSProp','GradientDescent','Momentum'],default='Adam')
parser.add_argument('--g_act', choices=['Sigmoid','Relu','Elu','Tanh',"Identity"], default = 'Sigmoid')
parser.add_argument('--f_act', choices=['Sigmoid','Relu','Elu','Tanh',"Identity"], default = 'Identity')

args = parser.parse_args()

data_name = 'ml-1m'
path = "./data/%s" % data_name + "/"
num_users = 6040
num_items = 3952
num_total_ratings = 1000209

train_ratio = 0.9
hidden_neuron = 500
random_seed = args.random_seed
batch_size = 256
lr = args.lr
train_epoch = args.train_epoch
optimizer_method = args.optimizer_method
#optimizer_method = 'RMSProp'
display_step = args.display_step
decay_epoch_step = 10
lambda_value = args.lambda_value

if args.f_act == "Sigmoid":
    f_act = tf.nn.sigmoid
elif args.f_act == "Relu":
    f_act = tf.nn.relu
elif args.f_act == "Tanh":
    f_act = tf.nn.tanh
elif args.f_act == "Identity":
    f_act = tf.identity
elif args.f_act == "Elu":
    f_act = tf.nn.elu
else:
    raise NotImplementedError("ERROR")

if args.g_act == "Sigmoid":
    g_act = tf.nn.sigmoid
elif args.g_act == "Relu":
    g_act = tf.nn.relu
elif args.g_act == "Tanh":
    g_act = tf.nn.tanh
elif args.g_act == "Identity":
    g_act = tf.identity
elif args.g_act == "Elu":
    g_act = tf.nn.elu
else:
    raise NotImplementedError("ERROR")

date = "0325"
result_path = './results/' + data_name + '/' + date + '/' + str(random_seed) + '_' + str(optimizer_method) + '_' + str(lr) + "_" +  str(current_time)+"/"




R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,\
user_train_set,item_train_set,user_test_set,item_test_set \
    = read_rating(path, num_users, num_items,num_total_ratings, P, 1, 0, train_ratio,random_seed)

with tf.Session() as sess:
    AutoRec = AutoRec(sess,args,
                         num_users,num_items,hidden_neuron,f_act,g_act,
                         R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                         train_epoch,batch_size, lr, optimizer_method, display_step, random_seed,
                         decay_epoch_step,lambda_value,
                         user_train_set, item_train_set, user_test_set, item_test_set,
                         result_path,date,data_name)

    AutoRec.run()