import matplotlib.pyplot as plt
from absl import app

# data of sh in case unchange speed
suXbar = ['t1', 't2', 't3', 't4', 't5', 't6']
suData = [19.5, 17, 16.7, 17.8, 21.7, 22]
suSys = [18.6, 17.6, 16.1, 19.5, 21.3, 23.3]

# data of sh in case unchange speed
buXbar = ['t1', 't2', 't3', 't4', 't5', 't6', 't7']
buData = [21.2, 23.4, 19.4, 31, 27, 16, 17.8]
buSys = [22.1, 25.6, 21.8, 34.2, 27.5, 17.6, 19]
def main(_argv):
    plt.plot(suXbar, suData, 'go-', label='data')
    plt.plot(suXbar, suSys, 'ro-', label='system')
    # plt.plot(buXbar, buData, 'go-', label='data')
    # plt.plot(buXbar, buSys, 'ro-', label='system')
    plt.title('TĐTB thực tế và của hệ thống trong trường hợp đi đều')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass