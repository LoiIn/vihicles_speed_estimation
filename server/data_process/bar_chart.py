from turtle import color
import matplotlib.pyplot as plt
from absl import app

Y = [11, 9, 7, 2]
X = ["0-1", "1-2", "2-3", ">3"]
data_label = ["11", "9", "7", "2"]


def main(_argv):
    plt.bar(X, Y, width=0.5)
    # plt.text(label)
    plt.xlabel('Khoảng')
    plt.ylabel('Số lượng')

    for i in range(len(X)): # your number of bars
        plt.text(
        x = X[i], #takes your x values as horizontal positioning argument 
        y = Y[i] + 0.5, #takes your y values as vertical positioning argument 
        s = data_label[i], # the labels you want to add to the data
        size = 11)

    plt.ylim(0, 15)
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass