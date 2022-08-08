import matplotlib.pyplot as plt
from absl import app

# data sh in vu4 , id = 1
# suXbar = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10']
# suData = [19.7, 20.7, 21.6, 20.9, 21.7, 21.4, 22.1, 20.2, 21.6, 21.6]
# suModel = [22, 22, 22, 22, 22, 22, 22, 22, 21.5, 21]
# data blade vu4, id = 5
# buXbar = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']
# buData = [22.7, 22.4, 20.9, 20.4, 20, 18.2, 18.8, 17.6, 16.7]
# buModel = [21, 19, 19, 18.5, 18, 18, 17.5, 17, 17]


# data sh in vf1 , id = 5
suXbar = ['t1', 't2', 't3', 't4', 't5', 't6', 't7']
suData = [46.2, 47.1, 48.8, 49.6, 50.6, 51, 50.5]
suModel = [47, 49, 49, 49, 49.5, 50, 50]

# data blade vu4, id = 4
buXbar = ['t1', 't2', 't3', 't4', 't5', 't6', 't7']
buData = [33, 34.1, 35.3, 36.3, 37, 37, 36.5]
buModel = [31, 32.5, 33, 33, 33, 34.5, 35]


# data sh in vs4 , id = 2
suXbar = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10']
suData = [29.8, 29.5, 28.7, 28, 27.8, 24.3, 24.2, 25.6, 24.1, 22]
suModel = [31, 31, 28.5, 28, 27.5, 25.5, 25, 25, 22.5, 22]
# data blade vu4, id = 5
buXbar = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8']
buData = [28.5, 27.8, 25.8, 25.4, 23.6, 22.2, 20.9, 19.4]
buModel = [30, 29, 27, 26, 25, 23, 23, 20.5]


def main(_argv):
    # plt.plot(suXbar, suData, 'go-', label='Model')
    # plt.plot(suXbar, suModel, 'r*-', label='Data')
    plt.plot(buXbar, buData, 'go-', label='Model')
    plt.plot(buXbar, buModel, 'r*-', label='Data')
    # plt.title('Trạng thái của xe 2')
    plt.xlabel('Thời điểm')
    plt.ylabel('Tốc độ (km/h)')
    # plt.xlim(0,2)
    # plt.ylim(0, 50)
    plt.ylim(15, 35)
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass