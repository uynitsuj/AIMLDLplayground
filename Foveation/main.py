from foveate_image import *
from matplotlib import pyplot as plt
from PIL import Image
import time
# import numpy

def main():
    
    # img = open('DSC_1625.jpg')
    img = Image.open('dog_in_me.jpeg')

    w, h = img.size
    print("Original Image Size", w, h)
    print("Pixel Count", w*h)

    img = np.array(list(img.getdata()))
    img = img.reshape((h,w,3))
    
    # print(img)

    # plt.imshow(img)
    # plt.show()

    fimg = FoveateImage(w, h)
    start = time.time()
    foveatedimg, idxs, coords_fc = fimg.foveate(img)
    elapsed = (time.time() - start)
    print(f"Elapsed time: {elapsed}(s)")
    print(f"Frequency: {1/elapsed}(fps)")
    print("Foveated Pixel Count", foveatedimg.shape)

    recon = torch.zeros((h,w,3), dtype=foveatedimg.dtype)
    # import pdb; pdb.set_trace()
    recon.view(-1, 3)[idxs] = foveatedimg[range(len(foveatedimg))]


    # x = torch.linspace(0, w - 1, w)
    # y = torch.linspace(0, h - 1, h)
    # x, y = torch.meshgrid(x, y)
    # x_centered = (x - w/2).transpose(1,0).contiguous()
    # y_centered = (y - h/2).transpose(1,0).contiguous()
    
    # xs = x_centered.view(-1)[idxs]
    # ys = y_centered.view(-1)[idxs]
    # ar = torch.arange(h*w)

    import pdb; pdb.set_trace()

    print(f"Compression: {len(idxs)/(w*h)*100}%")

    plt.imshow(recon)
    plt.show()


if __name__ == "__main__":
    main()
