import matplotlib.pyplot as plt

'''
:: default para: aggregator:sum, neighbor_sample_size:8, dim:16, n_iter:1
:: exp1: different aggregate mode: --aggregator=['sum', 'concat', 'neighbor']
:: exp2: neighbor_sample_size=[2,4,8,16,32]
:: exp3: dim=[8,16,32,64]
:: exp4: n_iter=[1,2,3,4]
'''


def get_paths(exp='exp1'):
    if exp=='exp1':
        paths = ['exp1_aggregator_'+i+'.log' for i in ['sum', 'concat', 'neighbor']]
    elif exp=='exp2':
        paths = ['exp2_neighbor_size_'+str(i)+'.log' for i in [2,4,8,16,32]]
    elif exp=='exp3':
        paths = ['exp3_dim_'+str(i)+'.log' for i in [8,16,32,64]]
    elif exp=='exp4':
        paths = ['exp4_n_iter_'+str(i)+'.log' for i in [1,2,3,4]]
    else:
        assert(False)
    return paths


def extra_data(paths):
    train_loss_list = []
    test_loss_list = []
    acc_list = []
    time_list = []
    for path in paths:
        with open(path) as f:
            lines = f.readlines()
            train_loss = []
            test_loss = []
            acc = []
            time = None
            for l in lines:
                if l.count('train_loss:'):
                    train_loss.append(float(l.split('train_loss:')[1].split(' ')[0]))
                    test_loss.append(float(l.split('test_loss:')[1].split(' ')[0]))
                    acc.append(float(l.split('acc:')[1].strip()))
                elif l.count('training time:'):
                    time = float(l.split('training time:')[1].strip())
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            acc_list.append(acc)
            time_list.append(time)
    return train_loss_list, test_loss_list, acc_list, time_list


def draw_exp1():
    train_loss_list, test_loss_list, acc_list, time_list = extra_data(get_paths('exp1'))
    aggregator = ['sum', 'concat', 'neighbor']
    nums = len(train_loss_list)
    epochs = range(len(train_loss_list[0]))
    
    plt.figure(figsize=(10 * scale,5 * scale))
    # draw train loss
    plt.subplot(1, 3, 1)
    for i in range(nums):
        plt.plot(epochs, train_loss_list[i], color=colors[i], label='agg='+str(aggregator[i]))
    plt.legend()
    plt.grid(True)
    plt.title("Average Training Loss during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")

    # draw test loss
    plt.subplot(1, 3, 2)
    for i in range(nums):
        plt.plot(epochs, test_loss_list[i], color=colors[i], label='agg='+str(aggregator[i]))
    plt.legend()
    plt.grid(True)
    plt.title("Test Loss during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")

    # draw acc
    plt.subplot(1, 3, 3)
    for i in range(nums):
        plt.plot(epochs, acc_list[i], color=colors[i], label='agg='+str(aggregator[i]))
    plt.legend()
    plt.grid(True)
    plt.title("Test Accuracy during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")

    plt.tight_layout()
    plt.savefig('exp1.png')
    # plt.show()


def draw_exp2():
    train_loss_list, test_loss_list, acc_list, time_list = extra_data(get_paths('exp2'))
    neighbor_sample_size=[2,4,8,16,32]
    nums = len(train_loss_list)
    epochs = range(len(train_loss_list[0]))

    plt.figure(figsize=(10 * scale,5 * scale))
    # draw train loss
    plt.subplot(1, 3, 1)
    for i in range(nums):
        plt.plot(epochs, train_loss_list[i], color=colors[i], label='neighbor='+str(neighbor_sample_size[i]))
    plt.legend()
    plt.grid(True)
    plt.title("Average Training Loss during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")

    # draw test loss
    plt.subplot(1, 3, 2)
    for i in range(nums):
        plt.plot(epochs, test_loss_list[i], color=colors[i], label='neighbor='+str(neighbor_sample_size[i]))
    plt.legend()
    plt.grid(True)
    plt.title("Test Loss during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")

    # draw acc
    plt.subplot(1, 3, 3)
    for i in range(nums):
        plt.plot(epochs, acc_list[i], color=colors[i], label='neighbor='+str(neighbor_sample_size[i]))
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title("Test Accuracy during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")

    plt.tight_layout()
    plt.savefig('exp2.png')
    # plt.show()


def draw_exp3():
    train_loss_list, test_loss_list, acc_list, time_list = extra_data(get_paths('exp3'))
    dim = [8,16,32,64]
    nums = len(train_loss_list)
    epochs = range(len(train_loss_list[0]))

    plt.figure(figsize=(8 * scale,6 * scale))
    # draw train loss
    plt.subplot(2, 2, 1)
    for i in range(nums):
        plt.plot(epochs, train_loss_list[i], color=colors[i], label='d='+str(dim[i]))
    plt.legend()
    plt.grid(True)
    plt.title("Average Training Loss during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")

    # draw test loss
    plt.subplot(2, 2, 2)
    for i in range(nums):
        plt.plot(epochs, test_loss_list[i], color=colors[i], label='d='+str(dim[i]))
    plt.legend()
    plt.grid(True)
    plt.title("Test Loss during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")

    # draw acc
    plt.subplot(2, 2, 3)
    for i in range(nums):
        plt.plot(epochs, acc_list[i], color=colors[i], label='d='+str(dim[i]))
    plt.legend()
    plt.grid(True)
    plt.title("Test Accuracy during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")

    # draw time
    plt.subplot(2, 2, 4)
    plt.plot(['d='+str(i) for i in dim], time_list, 'r*-')
    plt.grid(True)
    plt.title("Running Time required for Training.")
    plt.xlabel("d: Dimension of Embeddings")
    plt.ylabel("Training Time")

    plt.tight_layout()
    plt.savefig('exp3.png')
    # plt.show()


def draw_exp4():
    train_loss_list, test_loss_list, acc_list, time_list = extra_data(get_paths('exp4'))
    n_iter = [1,2,3,4]
    nums = len(train_loss_list)
    epochs = range(len(train_loss_list[0]))

    plt.figure(figsize=(8 * scale,6 * scale))
    # draw train loss
    plt.subplot(2, 2, 1)
    for i in range(nums):
        plt.plot(epochs, train_loss_list[i], color=colors[i], label='H='+str(n_iter[i]))
    plt.legend()
    plt.grid(True)
    plt.title("Average Training Loss during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")

    # draw test loss
    plt.subplot(2, 2, 2)
    for i in range(nums):
        plt.plot(epochs, test_loss_list[i], color=colors[i], label='H='+str(n_iter[i]))
    plt.legend()
    plt.grid(True)
    plt.title("Test Loss during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")

    # draw acc
    plt.subplot(2, 2, 3)
    for i in range(nums):
        plt.plot(epochs, acc_list[i], color=colors[i], label='H='+str(n_iter[i]))
    plt.legend()
    plt.grid(True)
    plt.title("Test Accuracy during Training.")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")

    # draw time
    plt.subplot(2, 2, 4)
    plt.plot(['H='+str(i) for i in n_iter], time_list, 'r*-')
    plt.grid(True)
    plt.title("Running Time required for Training.")
    plt.xlabel("H: Depth of Receptive Field")
    plt.ylabel("Training Time")

    plt.tight_layout()
    plt.savefig('exp4.png')
    # plt.show()


scale = 1.
colors = ['r', 'b', 'g', 'y', 'm']
draw_exp1()
draw_exp2()
draw_exp3()
draw_exp4()
