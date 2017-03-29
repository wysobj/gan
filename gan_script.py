import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

img_width = 28
img_height = 28
img_channel = 1
z_dim = 100
train_task = False
model_path = "generator.npz"

def generator_fc_layer(x, hdim, activation_func="relu", dropout=1.0):
    ac_funcs = {"relu" : tf.nn.relu, "sigmoid" : tf.nn.sigmoid, "tanh" : tf.nn.tanh, "none" : lambda x : x}
    func = ac_funcs.get(activation_func, None)
    if func == None:
        raise ValueError("Unsupported activation function.")
    xdim = x.get_shape().as_list()[-1]
    Wi = tf.Variable(tf.random_normal([xdim, hdim], mean=0., stddev=0.01), dtype=tf.float32)
    bi = tf.Variable(np.zeros(hdim, dtype="float32"), dtype=tf.float32)
    h = tf.matmul(x, Wi) + bi
    a = func(h)
    return a, Wi, bi

def discriminator_fc_layer(xg,  xd, hdim, activation_func="relu", dropout=1.0):
    ac_funcs = {"relu" : tf.nn.relu, "sigmoid" : tf.nn.sigmoid, "tanh" : tf.nn.tanh, "none" : lambda x : x}
    func = ac_funcs.get(activation_func, ac_funcs["none"])
    if func == None:
        raise ValueError("Unsupported activation function.")
    xdim = xg.get_shape().as_list()[-1]
    Wi = tf.Variable(tf.random_normal([xdim, hdim], mean=0., stddev=0.01), dtype=tf.float32)
    bi = tf.Variable(np.zeros(hdim, dtype="float32"), dtype=tf.float32)
    h_g = tf.matmul(xg, Wi) + bi
    h_d = tf.matmul(xd, Wi) + bi
    a_g = tf.nn.dropout(func(h_g), dropout)
    a_d = tf.nn.dropout(func(h_d), dropout)
    return a_g, a_d, Wi, bi

def build_generator(z_input):
    g_1, wg_1, bg_1 = generator_fc_layer(z_input, 150, activation_func="relu")
    g_2, wg_2, bg_2 = generator_fc_layer(g_1, 300, activation_func="relu")
    g_3, wg_3, bg_3 = generator_fc_layer(g_2, img_width * img_height * img_channel, activation_func="sigmoid")
    params = {"w1" : wg_1, "b1" : bg_1, "w2" : wg_2, "b2" : bg_2, "w3" : wg_3, "b3" : bg_3}
    output = g_3
    return output, params

def build_discriminator(g_input, d_input):
    dg_1, dd_1, wd_1, bd_1 = discriminator_fc_layer(g_input, d_input, 300, activation_func="relu", dropout=0.7)
    dg_2, dd_2, wd_2, bd_2 = discriminator_fc_layer(dg_1, dd_1, 150, activation_func="relu", dropout=0.7)
    dg_3, dd_3, wd_3, bd_3 = discriminator_fc_layer(dg_2, dd_2, 1, activation_func="none")
    params = {"w1" : wd_1, "b1" : bd_1, "w2" : wd_2, "b2" : bd_2, "w3" : wd_3, "b3" : bd_3}
    g_predict = dg_3
    d_predict = dd_3
    return g_predict, d_predict, params

def save_generator_params(path, params):
    np.savez(path, **params)

def restore_generator_params(path, tensors):
    params = np.load(path)
    if set(params.keys()) != tensors.keys():
        raise ValueError("Loaded parameters does not match the rebuilt model.")
    assigns = []
    for k, v in tensors.items():
        assigns.append(v.assign(params[k]))
    return assigns

def  print_costs(g_costs, d_costs):
    epochs = len(g_costs)
    x = np.arange(epochs)
    plt.plot(x, g_costs, label="Generator Cost")
    plt.plot(x, d_costs, label="Discriminator Cost")
    plt.legend(loc="upper right", labels=["Generator Cost", "Discriminator Cost"])
    plt.show()


def train():
    k_steps = 1
    batch_size = 256
    max_epochs = 200
    lr = 0.001
    momentum = 0.9
    use_gpu = True
    save_imgs_batch = 100
    imgs_path = "imgs.npz"

    # load data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    z_input = tf.placeholder(tf.float32, shape=[None, z_dim])
    d_input = tf.placeholder(tf.float32, shape=[None, img_width * img_height * img_channel])

    g_output, g_params = build_generator(z_input)
    g_predict, d_predict, d_params = build_discriminator(g_output, d_input)

    d_real_entropy = tf.nn.sigmoid_cross_entropy_with_logits(d_predict, tf.ones_like(d_predict))
    d_fake_entropy = tf.nn.sigmoid_cross_entropy_with_logits(g_predict, tf.zeros_like(g_predict))
    g_entropy = tf.nn.sigmoid_cross_entropy_with_logits(g_predict, tf.ones_like(g_predict))

    d_cost = tf.reduce_mean(d_real_entropy + d_fake_entropy)
    g_cost = tf.reduce_mean(g_entropy)

    optimizer = tf.train.AdamOptimizer(lr)
    d_train = optimizer.minimize(d_cost, var_list=list(d_params.values()))
    g_train = optimizer.minimize(g_cost, var_list=list(g_params.values()))

    if not use_gpu:
        init = tf.global_variables_initializer()
    else:
        init = tf.initialize_all_variables()

    # save some generator results for visualization
    imgs = []
    img_save_num = 100

    # train procedure
    d_costs_trace = []
    g_costs_trace = []
    with tf.Session() as sess:
        sess.run(init)
        epoch_rounds = int(mnist.train.num_examples / batch_size)
        for epoch in range(max_epochs):
            for er in range(epoch_rounds):
                for ks in range(k_steps):
                    samples, _ = mnist.train.next_batch(batch_size)
                    noises = np.random.uniform(-1, 1, (batch_size, z_dim)).astype(np.float32)
                    sess.run(d_train, feed_dict={z_input : noises, d_input : samples})
                noises = np.random.uniform(-1, 1, (batch_size, z_dim)).astype(np.float32)
                sess.run(g_train, feed_dict={z_input : noises})
            noises_display = np.random.uniform(-1, 1, (batch_size, z_dim)).astype(np.float32)
            dc = sess.run(d_cost, feed_dict={z_input : noises_display, d_input : samples})
            gc = sess.run(g_cost, feed_dict={z_input : noises_display, d_input : samples})
            d_costs_trace.append(dc)
            g_costs_trace.append(gc)
            if dc.any() == np.nan:
                print("Bad training result.")
                break
            fake = sess.run(g_output, feed_dict={z_input : noises_display})
            print("epoch {} : d_cost {}, g_cost {}".format(epoch, dc, gc))
            imgs.append(fake[:img_save_num])
        model = {}
        for k, v in g_params.items():
            model[k] = sess.run(v)
    np.savez(imgs_path, imgs)
    save_generator_params(model_path, model)
    print_costs(g_costs_trace, d_costs_trace)


def test():
    img_raws = 10
    img_columns = 10
    img_num = img_raws * img_columns
    z_input = tf.placeholder(tf.float32, [img_num, z_dim])
    g_output, g_params = build_generator(z_input)
    assign = restore_generator_params(model_path, g_params)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(assign)
        noises = np.random.uniform(-1, 1, (img_num, z_dim)).astype(np.float32)
        imgs = sess.run(g_output, feed_dict={z_input : noises})
        
    # visualization
    for i in range(img_num):
        plt.subplot(img_raws, img_columns, i+1)
        plt.imshow(imgs[i].reshape((img_width, img_height)), cmap="gray")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    if train_task:
        print("Begin training...")
        train()
    else:
        test()