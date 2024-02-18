from foveate_image import *
from matplotlib import pyplot as plt
from PIL import Image
# import numpy

def main():
    # img = open('DSC_1625.jpg')
    img = Image.open('DSC_1625.jpg')

    w, h = img.size

    img = np.array(list(img.getdata()))
    img = img.reshape((h,w,3))
    
    # print(img)


    # plt.imshow(img)
    # plt.show()

    fimg = FoveatedImage()
    foveatedimg = fimg.foveate(img)


    plt.imshow(foveatedimg)
    plt.show()


if __name__ == "__main__":
    main()
