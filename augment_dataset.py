import imageio
from ms_truchet import MultiScaleTruchetPattern

N_SAMPLES = 1000
IMG_MODE = "L"

def main():
    how_many_tiles = 2
    # of_size = 48
    of_size = 24
    multiscaleTruchetTiling = MultiScaleTruchetPattern(how_many_tiles, of_size, 'white','black')
    for i in range(N_SAMPLES):
        img = multiscaleTruchetTiling.paint_a_multiscale_truchet()
        if img.mode != IMG_MODE:
            img = img.convert(IMG_MODE)
        imageio.imsave("imgs/train/"+str(i)+".gif", img)

if __name__ == '__main__':
    main()
