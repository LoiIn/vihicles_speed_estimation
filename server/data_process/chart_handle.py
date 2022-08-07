import matplotlib.pyplot as plt
from absl import app

# data of sh in case unchange speed
suXbar = ['t1', 't2', 't3', 't4', 't5', 't6']
suData = [20.4, 21.7, 22.3, 22, 23, 23.4]
suSys = [20, 21, 21, 22, 22, 22]

# data of balde in case unchange speed
buXbar = [ 't7', 't8', 't9', 't10', 't11', 't12']
buData = [ 26.2, 26.1, 25.8, 25.8, 25.3, 24.7]
buSys = [23, 23, 23, 23, 24, 24]
def main(_argv):
    plt.plot(suXbar, suData, 'go-', label='data')
    plt.plot(suXbar, suSys, 'ro-', label='system')
    plt.plot(buXbar, buData, 'go-', label='data')
    plt.plot(buXbar, buSys, 'ro-', label='system')
    plt.title('Trạng thái của xe 2')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass