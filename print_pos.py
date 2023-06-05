from matplotlib import pyplot as plt
import time, cv2


def mt_to_pix(x_mt, y_mt, IMG_SIZE, IMG_MT_SIZE):
    x = (x_mt * IMG_SIZE[1]) / IMG_MT_SIZE[0]
    y = (y_mt * IMG_SIZE[0]) / IMG_MT_SIZE[1]
    return x,y


def main():
    #Env1 - Reggio - Real Mt Dimensions e Env Path
    #IMG_MT_SIZE = [264, 133]
    #image_path = '/home/ubuntu/env.png'

    #Env2 - Roma - Real Mt Dimensions e Env Path
    IMG_MT_SIZE = [440, 262]
    image_path = '/home/ubuntu/env2.png'

    pos_path = '/home/ubuntu/pos.txt'

    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nr1 = int(image.shape[0] / 256.)
    nc1 = int(image.shape[1] / 256.)
    image_RGB = image_RGB[0:nr1*256, 0:nc1*256]
    IMG_SIZE = image_RGB.shape

    #plt.axis('off')
    plt.imshow(image_RGB)

    pos_x = []
    pos_y = []

    f = open(pos_path, "r")
    riga = f.readline()
    while riga != "":
        sx, sy = riga.split(' ')

        x, y = mt_to_pix(float(sx), float(sy), IMG_SIZE, IMG_MT_SIZE)
        
        plt.plot(x, IMG_SIZE[0]-y, color='green', marker='o', markersize=7)

        riga = f.readline()
    f.close()

    plt.show()

    return

main()