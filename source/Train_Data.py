

import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import random
import matplotlib.pyplot as plt

# result_dir contains plot, test error and training error
result_dir = '../RESULT'

def sigmoid(S):
    return 1 / (1 + np.exp(-S))

def softmax(S):
    A = np.exp(S)
    A /= A.sum(axis=1, keepdims=True)
    return A

def compute_nnet_outputs(Ws, X, need_all_layer_outputs):
    As = []
    Z = X
    for layer_idx in range(len(Ws) - 1):
        s = sigmoid(np.dot(Z,Ws[layer_idx]))
        Z = np.hstack((np.ones((len(s),1)),s))
        As.append(Z)
    As.append(softmax(np.dot(As[(len(Ws) - 2)],Ws[-1])))
    return As[-1] if need_all_layer_outputs == False else As

def draw_plot(train_errs,val_errs,file_name):
    num_epochs = len(train_errs)
    plt.figure(figsize=(14,10))

    plt.plot(range(num_epochs), train_errs, 'r', label='train err')
    plt.plot(range(num_epochs), val_errs, 'b--', label='val err')

    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)

    plt.xlabel('num epochs',fontsize=15)
    plt.ylabel('error',fontsize=15)
    plt.legend(loc='best',fontsize=25)
    plt.savefig(file_name)

def train_nnet(train_X, train_Y, val_X, val_Y, hid_layer_sizes, wd_level,mb_size, learning_rate, max_patience, max_epoch=1000000, momentum_param=0.):
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Init Ws
    K = len(np.unique(train_Y)) # Num classes
    layer_sizes = [train_X.shape[1] - 1] + hid_layer_sizes + [K]
    np.random.seed(0) # This will fix the randomization; so, you and me will have the same results
    Ws = [np.random.randn(layer_sizes[l]+1, layer_sizes[l+1]) / np.sqrt(layer_sizes[l]+1) 
          for l in range(len(layer_sizes)-1)]
   
    one_hot_Y = np.zeros((len(train_Y), K))
    one_hot_Y[np.arange(len(train_Y)), train_Y.reshape(-1)] = 1
    N = len(train_X) # Num training examples
    rnd_idxs = range(N) # Random indexes   
    
    train_errs = []
    val_errs = []
    V = [np.zeros(Ws[i].shape) for i in range(len(Ws))]
    
    patience = max_patience
    best_epoch_idx = 0
    best_val_err = 101 # Because the biggest error percentage can reach is 100 
    best_Ws = Ws
    # Train
    for epoch in range(max_epoch):
        np.random.shuffle(rnd_idxs)
        for start_idx in range(0, N, mb_size):
            # Get minibach
            mb_X = train_X[rnd_idxs[start_idx:start_idx+mb_size]]          
            mb_Y = one_hot_Y[rnd_idxs[start_idx:start_idx+mb_size]]
            
            # Forward-prop
            As = compute_nnet_outputs(Ws, mb_X, True)
            
            # Back-prop; on the way, compute each layer's gradient and update its W
            delta = As[-1] - mb_Y
            grad = (As[-2].T.dot(delta) / len(mb_X) + wd_level* 2*Ws[-1])
            V[-1] = momentum_param * V[-1] - learning_rate * grad 
            Ws[-1] += V[-1]
            
            for i in range(2, len(Ws) + 1):
                delta = delta.dot(Ws[-i + 1].T) * As[-i] * (1 - As[-i])
                delta = delta[:,1:] #remove bias
                if i == len(Ws):
                    grad = (mb_X.T.dot(delta)/len(mb_X) + wd_level* 2*Ws[-i])
                else:
                    grad =( As[-i-1].T.dot(delta) / len(mb_X) + wd_level* 2*Ws[-i])
                V[-i] = momentum_param*V[-i] - learning_rate * grad 
                Ws[-i] += V[-i]
        train_h = compute_nnet_outputs(Ws,train_X,need_all_layer_outputs=False)
        train_errs.append(np.mean(np.argmax(train_h, axis=1) != train_Y.squeeze()) * 100)
        val_h = compute_nnet_outputs(Ws,val_X,False)
        val_errs.append(np.mean(np.argmax(val_h, axis=1) != val_Y.squeeze()) * 100)

        if patience is not None:
            if val_errs[-1] < best_val_err:
                patience = max_patience
                best_val_err = val_errs[-1]
                best_Ws = Ws
                best_epoch_idx = epoch
            else:
                patience = patience -1 
                if patience == 0:
                    break
        if epoch % 100 == 0:
            if patience is not None:
                print 'Epoch: %d,training err: %f, val err: %f,patience: %d' %(epoch,train_errs[-1],val_errs[-1],patience)
            else:
                print 'Epoch: %d,training err: %f, val err: %f,patience: None' %(epoch,train_errs[-1],val_errs[-1])
    
    if patience is not None:
        s = 'Info of returned Ws: best epoch %d, train err %f, val err %f'  % (best_epoch_idx, train_errs[best_epoch_idx], val_errs[best_epoch_idx])
        print s  
        with open('{0}/{1}_{2}_{3}__Train.txt'.format(result_dir,hid_layer_sizes[0],learning_rate,max_patience), "w") as text_file:
            text_file.write(s) 
        Ws = best_Ws    
    else:
        s =  'Info of returned Ws: epoch %d, train err %f, val err %f' % (epoch, train_errs[-1], val_errs[-1])
        print s
        with open('{0}/{1}_{2}_{3}__Train.txt'.format(result_dir,hid_layer_sizes[0],learning_rate,max_patience), "w") as text_file:
            text_file.write(s) 
    
    df_err = pd.DataFrame({'train_errs':train_errs,'val_errs':val_errs})
    df_err.to_csv('{0}/{1}_{2}_{3}__ERR.csv'.format(result_dir,hid_layer_sizes[0],learning_rate,max_patience))

    file_plot_name = '{0}/{1}_{2}_{3}__Plot.png'.format(result_dir,hid_layer_sizes[0],learning_rate,max_patience)
    draw_plot(train_errs,val_errs,file_plot_name)
    return Ws, train_errs, val_errs

full_df = pd.DataFrame.from_csv('../DATA.csv')

# Choose all column except rank and asin
X = full_df.iloc[:,2:].values
Y = ((full_df['rank'].values - 1)/ 20).reshape(-1,1)

np.random.seed(0)
c = list(zip(X,Y))
np.random.shuffle(c)
a, b = zip(*c)
X = np.array(a)
Y = np.array(b)

train_X = X[:10500]
train_Y = Y[:10500]

X_mean = train_X.mean(axis=0)
X_std = train_X.std(axis=0)
train_X = (train_X - X_mean) / X_std
train_X = np.hstack([np.ones((len(train_X), 1)), train_X])

val_X = X[10500:(10500 + 3500)]
val_Y = Y[10500:(10500 + 3500)]
val_X = (val_X - X_mean) / X_std
val_X = np.hstack([np.ones((len(val_X), 1)), val_X])

test_X = X[(10500 + 3500):]
test_Y = Y[(10500 + 3500):]
test_X = (test_X - X_mean) / X_std
test_X = np.hstack([np.ones((len(test_X), 1)), test_X])

start = time.time()

hid_layer_sizes=[50]
learning_rate=1.0
max_patience=20000
Ws_0, train_errs_0, val_errs_0 = train_nnet(train_X, train_Y, val_X, val_Y, hid_layer_sizes=hid_layer_sizes, 
                                            wd_level=0.00, mb_size=len(train_X), learning_rate=learning_rate, 
                                            max_patience=max_patience, max_epoch=400000) 

final_h = compute_nnet_outputs(Ws_0,test_X,False)
test_error = np.mean(np.argmax(final_h, axis=1) != test_Y.squeeze()) * 100
# Write test error into file
with open('{0}/{1}_{2}_{3}__Test.txt'.format(result_dir,hid_layer_sizes[0],learning_rate,max_patience), "w") as text_file:
    text_file.write(str(test_error)) 
print 'Test error ',test_error

end = time.time()
print 'Time elapsed',(end - start)