import random
import math

def mae(target,predict):
    return abs(target - predict)
def mse(target,predict):
    return (target-predict)**2
def rmse(target,predict):
    return mse(target,predict)

num_samples = input("Nhap so luong samples: ")
if num_samples.isnumeric() :
    num_samples = int( num_samples )
else :
    print("number of samples must be an integer number")
    exit()

loss_name = input("Nhap loss_name: ")
final_result = 0
for i in range(num_samples):
    target = random.uniform(0,10)
    predict = random.uniform(0,10)
    if loss_name == "MAE":
        loss = mae(target,predict)
    elif loss_name == "MSE":
        loss = mse(target,predict)
    else:
        loss = rmse(target,predict)
    final_result += loss
    print("loss name: {}, ".format(loss_name),end="" )
    print("sample: {}, ".format(i),end="")
    print("pred: {}, ".format(predict),end="")
    print("target: {}, ".format(target),end="")
    print("loss: {}".format(loss))
if loss_name == "MAE" or loss_name == "MSE":
    print("Final {}: {}".format(loss_name,final_result/num_samples))
else:
    print("Final {}: {}".format(loss_name,math.sqrt(final_result/num_samples)))