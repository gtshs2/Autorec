import tensorflow as tf
import time
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class AutoRec():
    def __init__(self,sess,args,
                 num_users,num_items,hidden_neuron,f_act,g_act,
                 R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R,num_train_ratings,num_test_ratings,
                 train_epoch,batch_size,lr,optimizer_method,
                 display_step,random_seed,
                 decay_epoch_step,lambda_value,
                 user_train_set, item_train_set, user_test_set, item_test_set,
                 result_path,date,data_name):

        self.sess = sess
        self.args = args

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_neuron = hidden_neuron

        self.R = R
        self.mask_R = mask_R
        self.C = C
        self.train_R = train_R
        self.train_mask_R = train_mask_R
        self.test_R = test_R
        self.test_mask_R = test_mask_R
        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings

        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.num_batch = int(self.num_users / float(self.batch_size)) + 1

        self.lr = lr
        self.optimizer_method = optimizer_method
        self.display_step = display_step
        self.random_seed = random_seed

        self.f_act = f_act
        self.g_act = g_act

        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch

        self.lambda_value = lambda_value

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []

        self.user_train_set = user_train_set
        self.item_train_set = item_train_set
        self.user_test_set = user_test_set
        self.item_test_set = item_test_set

        self.result_path = result_path
        self.date = date
        self.data_name = data_name

    def run(self):
        self.prepare_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in xrange(self.train_epoch):
            self.train_model(epoch_itr)
            self.test_model(epoch_itr)
        self.make_records()

    def prepare_model(self):
        self.input_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R")
        self.input_mask_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_mask_R")

        V = tf.get_variable(name="V", initializer=tf.truncated_normal(shape=[self.num_items, self.hidden_neuron],
                                         mean=0, stddev=0.03),dtype=tf.float32)
        W = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[self.hidden_neuron, self.num_items],
                                         mean=0, stddev=0.03),dtype=tf.float32)

        mu = tf.get_variable(name="mu", initializer=tf.zeros(shape=self.hidden_neuron),dtype=tf.float32)
        b = tf.get_variable(name="b", initializer=tf.zeros(shape=self.num_items), dtype=tf.float32)

        pre_Encoder = tf.matmul(self.input_R,V) + mu


        self.Encoder = self.g_act(pre_Encoder)
        pre_Decoder = tf.matmul(self.Encoder,W) + b
        self.Decoder = self.f_act(pre_Decoder)


        pre_cost1 = tf.multiply((self.input_R - self.Decoder) , self.input_mask_R)
        cost1 = tf.square(self.l2_norm(pre_cost1))
        pre_cost2 = tf.square(self.l2_norm(W)) + tf.square(self.l2_norm(V))
        cost2 = self.lambda_value * 0.5 * pre_cost2

        self.cost = cost1 + cost2

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "Adadelta":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "Adagrad":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optimizer_method == "GradientDescent":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.optimizer_method == "Momentum":
            optimizer = tf.train.MomentumOptimizer(self.lr,0.9)
        else:
            raise ValueError("Optimizer Key ERROR")

        gvs = optimizer.compute_gradients(self.cost)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    def train_model(self,itr):
        start_time = time.time()
        random_perm_doc_idx = np.random.permutation(self.num_users)

        batch_cost = 0
        for i in xrange(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size : (i+1) * self.batch_size]

            _, Cost = self.sess.run(
                [self.optimizer, self.cost],
                feed_dict={self.input_R: self.train_R[batch_set_idx, :],
                           self.input_mask_R: self.train_mask_R[batch_set_idx, :]})

            batch_cost = batch_cost + Cost
        self.train_cost_list.append(batch_cost)

        #if itr % self.display_step == 0:
        print ("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(batch_cost),
               "Elapsed time : %d sec" % (time.time() - start_time))

    def test_model(self,itr):
        start_time = time.time()
        Cost,Decoder = self.sess.run(
            [self.cost,self.Decoder],
            feed_dict={self.input_R: self.test_R,
                       self.input_mask_R: self.test_mask_R})

        self.test_cost_list.append(Cost)

        if itr % self.display_step == 0:
            Estimated_R = Decoder.clip(min=1, max=5)
            unseen_user_test_list = list(self.user_test_set - self.user_train_set)
            unseen_item_test_list = list(self.item_test_set - self.item_train_set)

            for user in unseen_user_test_list:
                for item in unseen_item_test_list:
                    if self.test_mask_R[user,item] == 1: # exist in test set
                        Estimated_R[user,item] = 3

            pre_numerator = np.multiply((Estimated_R - self.test_R), self.test_mask_R)
            numerator = np.sum(np.square(pre_numerator))
            denominator = self.num_test_ratings
            RMSE = np.sqrt(numerator / float(denominator))

            self.test_rmse_list.append(RMSE)

            print ("Testing //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(Cost), " RMSE = {:.5f}".format(RMSE),
                   "Elapsed time : %d sec" % (time.time() - start_time))
            print "=" * 100

    def make_records(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        overview = './results/'+str(self.data_name)+'/'+str(self.date)+'/overview.txt'
        basic_info = self.result_path + "basic_info.txt"
        train_record = self.result_path + "train_record.txt"
        test_record = self.result_path + "test_record.txt"

        with open (train_record,'w') as f:
            f.write(str("Cost:"))
            f.write('\t')
            for itr in range(len(self.train_cost_list)):
                f.write(str(self.train_cost_list[itr]))
                f.write('\t')
            f.write('\n')

        with open (test_record,'w') as g:
            g.write(str("Cost:"))
            g.write('\t')
            for itr in range(len(self.test_cost_list)):
                g.write(str(self.test_cost_list[itr]))
                g.write('\t')
            g.write('\n')

            g.write(str("RMSE:"))
            for itr in range(len(self.test_rmse_list)):
                g.write(str(self.test_rmse_list[itr]))
                g.write('\t')
            g.write('\n')

        with open(basic_info,'w') as h:
            h.write(str(self.args))

        with open(overview,'a') as f:
            f.write(str(self.random_seed))
            f.write('\t')
            f.write(str(self.optimizer_method))
            f.write('\t')
            f.write(str(self.lr))
            f.write('\t')
            f.write(str(self.train_cost_list[-1]))
            f.write('\t')
            f.write(str(self.test_cost_list[-1]))
            f.write('\t')
            f.write(str(self.test_rmse_list[-1]))
            f.write('\n')

        Train = plt.plot(self.test_cost_list,label='Train')
        Test = plt.plot(self.test_cost_list,label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.legend()
        plt.savefig(self.result_path+"Cost.png")
        plt.clf()

        Test = plt.plot(self.test_rmse_list,label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(self.result_path+"RMSE.png")
        plt.clf()

    def l2_norm(self,tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))



