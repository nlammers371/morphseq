import sys
import skimage.io
import numpy as np
import glob
import os
import concurrent.futures



def findnth(haystack, needle, n):
    parts = haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)


im_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/20230531/bf_timeseries_stack0850_pitch040/W001/P00001/T0004/wt_W001_P00001_T0004_Z005_CH1.tif"
fulldata = open(im_path, 'rb').read()
metadata = fulldata.partition(b'<Data>')[2].partition(b'</Data>')[0].decode()

# <Width Type="System.Int32">3623441</Width>\r\n        <Height Type="System.Int32">2717581</Height>\r\n
# param_string = 'ShootingDateTime'
# param_string = 'StageLocationX'
# param_string = 'LensName'
# param_string = 'Observation Type'
param_string = 'Width'

if (param_string == 'Width') or (param_string == 'Height'):
    ind1 = findnth(metadata, param_string + ' Type', 2)
    ind2 = findnth(metadata, '/' + param_string, 2)
else:
    ind1 = metadata.find(param_string)
    ind2 = metadata.find('/' + param_string)
long_string = metadata[ind1:ind2]
subind1 = long_string.find(">")
subind2 = long_string.find("<")
param_val = long_string[subind1+1:subind2]

sysind = long_string.find("System.")
dtype = long_string[sysind+7:subind1-1]
if 'Int' in dtype:
    param_val = int(param_val)

if param_string == "ShootingDateTime":
    param_val = param_val / 10 / 1000 / 1000  # convert to seconds (native unit is 100 nanoseconds)


def get_position_keyence(image, path):
    fulldata = open(path, 'rb').read()
    text = fulldata.partition(b'<Data>')[2].partition(b'</Data>')[0].decode()
    text = text.partition('<XyStageRegion')[2]
    x = int(text.partition('<X Type="System.Int32">')[2].partition('</X>')[0])
    y = int(text.partition('<Y Type="System.Int32">')[2].partition('</Y>')[0])
    width = int(text.partition('<Width Type="System.Int32">')[2].partition('</Width>')[0])
    height = int(text.partition('<Height Type="System.Int32">')[2].partition('</Height>')[0])
    return -y, -x, height, width


def get_grid_position_keyence(image, params, path):
    pos_name = glob.glob(os.path.dirname(path) + '/_X*Y*')[0]
    x, y = (int(pos_name[-3:])), (int(pos_name[-7:-4]))
    return x, y, 1, 1


class ImageLoader:
    def __init__(self, pattern=None):  # 't{time:d}_{strength}0{drug:d}_s{section:d}_w{layer:d}.TIF'):
        self.pattern = pattern
        if self.pattern is not None:
            self.load()

    def load(self):
        wildcard_pattern = re.sub('\{[^\}]*\}', '*', self.pattern)
        print(wildcard_pattern)
        self.paths = sorted(glob.glob(wildcard_pattern))
        self.param_ranges = {}
        for path in self.paths:
            params = self.to_params(path)
            for name in params:
                if name not in self.param_ranges:
                    self.param_ranges[name] = [params[name]]
                elif params[name] not in self.param_ranges[name]:
                    self.param_ranges[name].append(params[name])

        for name in self.param_ranges:
            self.param_ranges[name].sort()

    def to_path(self, params):
        path = self.pattern.format_map(params)
        return path

    def make_path(self, params):
        path = self.to_path(params)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def to_params(self, path):
        result = parse.parse(self.pattern, path, case_sensitive=True)
        return None if result == None else result.named

    def all_params(self, **params):
        all_params = []
        for name in self.param_ranges:
            if name not in params:
                for val in self.param_ranges[name]:
                    new_params = params.copy()
                    new_params[name] = val
                    all_params += self.all_params(**new_params)
                break
        else:
            for name in params:
                if type(params[name]) != str and (
                        hasattr(params[name], '__iter__') or hasattr(params[name], '__getitem__')):
                    values = params[name]
                    for val in values:
                        params[name] = val
                        all_params += self.all_params(**params)
                    break
            else:
                return [params]
        return all_params

    def load_params(self, **params):
        params = self.all_params(**params)
        images = [skimage.io.imread(self.to_path(param)) for param in params]
        return images, params

    def load_param_stack(self, stack_param=None, **params):
        params = self.all_params(**params)
        images = [skimage.io.imread(self.to_path(param)) for param in params]
        if stack_param != None:
            if stack_param in self.param_ranges and stack_param not in params:
                new_images_list = []
                for cur_param in self.param_ranges[stack_param]:
                    new_images_list.append(
                        [image for image, param in zip(images, params) if param[stack_param] == cur_param])
                images = [np.stack(image_list, axis=-1) for image_list in zip(*new_images_list)]
                params = [param for param in params if param[stack_param] == self.param_ranges[stack_param][0]]
        return images, params


def downscale_mean(image, factor):
    image = image[:image.shape[0] // factor * factor, :image.shape[1] // factor * factor]
    image = image.reshape(image.shape[0] // factor, factor, image.shape[1] // factor, factor, *image.shape[2:])
    return np.mean(image, axis=(1, 3)).astype(image.dtype)


def illum_correction(images):
    images = np.array(images)
    background = np.min(images, axis=0)
    # signal = np.percentile(images, axis=0).astype(images[0].dtype) - background
    images = (images - background)
    signal_val = np.percentile(images, 99.9, axis=(0, 1, 2))
    images = images / signal_val
    images[images < 0] = 0
    images[images > 1] = 1
    return images


class CompositeImage:
    def __init__(self):
        self.image = None
        self.x = None
        self.y = None
        self.width = None
        self.height = None

    def xres(self):
        if self.image is None:
            return 1
        return self.image.shape[0] / self.width

    def yres(self):
        if self.image is None:
            return 1
        return self.image.shape[1] / self.height

    def add_image(self, newimage, x, y, width=None, height=None):
        # print ('adding image', newimage.shape, x, y, width, height, self.x, self.y, self.width, self.height, self.xres(), self.yres())
        if width is None:
            width = newimage.shape[0] / self.xres()
        if height is None:
            height = newimage.shape[1] / self.yres()

        if (self.image is None):
            self.image = newimage
            self.mask = np.ones(newimage.shape, dtype='uint8')
            self.x, self.y, self.width, self.height = x, y, width, height
        else:
            if abs(newimage.shape[0] / width - self.xres()) > 1 / self.xres() or abs(
                    newimage.shape[1] / height - self.yres()) > 1 / self.yres():
                print(newimage.shape[0] / width, self.xres(), newimage.shape[1] / height, self.yres())
                newimage, width, height = self.rescale(newimage, width, height)
            pix_x, pix_y = int((x - self.x) * self.xres()), int((y - self.y) * self.yres())

            if pix_x < 0 or pix_x + newimage.shape[0] > self.image.shape[0] or pix_y < 0 or pix_y + newimage.shape[1] > \
                    self.image.shape[1]:
                newshape = (max(self.image.shape[0] + max(0, -pix_x), pix_x + newimage.shape[0]),
                            max(self.image.shape[1] + max(0, -pix_y), pix_y + newimage.shape[1]), *self.image.shape[2:])
                newrootimage = np.zeros(newshape, dtype=self.image.dtype)
                newrootmask = np.zeros(newshape, dtype='uint8')
                negpix_x, negpix_y = max(0, -pix_x), max(0, -pix_y)
                newrootimage[negpix_x:negpix_x + self.image.shape[0],
                negpix_y:negpix_y + self.image.shape[1]] = self.image
                newrootmask[negpix_x:negpix_x + self.image.shape[0],
                negpix_y:negpix_y + self.image.shape[1]] = self.mask

                self.x = min(self.x, x)
                self.y = min(self.y, y)
                self.width = newrootimage.shape[0] / self.xres()
                self.height = newrootimage.shape[1] / self.yres()

                self.image = newrootimage
                self.mask = newrootmask

                pix_x, pix_y = int((x - self.x) * self.xres()), int((y - self.y) * self.yres())

            x0, x1, y0, y1 = pix_x, pix_x + newimage.shape[0], pix_y, pix_y + newimage.shape[1]
            # mask = self.mask[x0:x1,y0:y1]
            # self.image[x0:x1,y0:y1][mask!=0] //= 2
            # self.image[x0:x1,y0:y1][mask!=0] = (self.image[x0:x1,y0:y1][mask!=0].astype(int) * mask[mask!=0] // (mask[mask!=0]+1)).astype('uint8')
            self.image[x0:x1, y0:y1] = newimage
            self.mask[x0:x1, y0:y1] = 1
            # self.image[pix_x:pix_x+newimage.shape[0], pix_y:pix_y+newimage.shape[1]] = newimage
            # self.image[pix_x:pix_x+newimage.shape[0], pix_y:pix_y+newimage.shape[1]] += newimage
            # self.mask[pix_x:pix_x+newimage.shape[0], pix_y:pix_y+newimage.shape[1]] += 1

    def full_image(self, fill_val=0):
        full_image = self.image.copy()
        full_image[self.mask == 0] = fill_val
        return full_image

    def best_position(self, image, start_x, start_y, width=None, height=None, search_radius=25, grow_image=False):
        if grow_image:
            newimage = np.zeros((image.shape[0] + 2, image.shape[1] + 2, *image.shape[2:]), dtype=image.dtype)
            newimage[1:-1, 1:-1] = image
            newimage[0, 1:-1] = image[0, :]
            newimage[-1, 1:-1] = image[-1, :]
            newimage[1:-1, 0] = image[:, 0]
            newimage[1:-1, -1] = image[:, -1]
            start_x -= 1 / self.xres()
            start_y -= 1 / self.yres()
            image = newimage

        if width is None:
            width = image.shape[0] / self.xres()
        if height is None:
            height = image.shape[1] / self.yres()

        start_pix_x, start_pix_y = int((start_x - self.x) * self.xres()), int((start_y - self.y) * self.yres())

        if type(search_radius) == int:
            search_radius = search_radius, search_radius

        best_score = 0
        best_pos = 0, 0

        for xoff in range(-search_radius[0], search_radius[0] + 1):
            for yoff in range(-search_radius[1], search_radius[1] + 1):
                pix_x, pix_y = start_pix_x + xoff, start_pix_y + yoff

                img_x, img_y = max(0, -pix_x), max(0, -pix_y)
                pix_x, pix_y = max(0, pix_x), max(0, pix_y)
                width, height = max(0, image.shape[0] - img_x), max(0, image.shape[1] - img_y)
                # width = min(self.image.shape[0], pix_x+width) - self.image.shape[0] + width
                # height = min(self.image.shape[1], pix_y+height) - self.image.shape[1] + height

                maskslice = self.mask[pix_x:pix_x + width, pix_y:pix_y + height]
                rootslice = self.image[pix_x:pix_x + width, pix_y:pix_y + height]
                width, height = rootslice.shape[:2]
                imageslice = image[img_x:img_x + width, img_y:img_y + height]
                # print (imageslice.shape, maskslice.shape, rootslice.shape, image.shape, width, height, img_x, img_y, pix_x, pix_y)
                if np.sum(maskslice) > 0:
                    score = np.sum(rootslice[maskslice] * imageslice[maskslice]) / np.sum(maskslice)
                    if score > best_score:
                        best_score = score
                        best_pos = xoff, yoff

        if grow_image:
            return (start_pix_x + best_pos[0] + 1) / self.xres(), (start_pix_y + best_pos[1] + 1) / self.yres()
        return best_pos[0] / self.xres(), best_pos[1] / self.yres()


def stitch_keyence_all(dirpath):
    imageloader = ImageLoader()
    imageloader.image_path = dirpath
    if dirpath.count('8-28') > 0:
        imageloader.file_pattern = "Base{base:d}/W{well:03d}/P{part:05d}/{date}_{experiment}_B{base:d}_W{well:03d}_P{part:05d}_CH{layer:d}.tif"
    else:
        imageloader.file_pattern = "Base{base:d}/W{well:03d}/P{part:05d}/{date}_{experiment}W{well:03d}_P{part:05d}_CH{layer:d}.tif"
    imageloader.load()

    imagesaver = ImageLoader()
    imagesaver.image_path = 'fisseq_images/stitched/'
    imagesaver.file_pattern = "{resolution}/{date}_{method}/Base{base:d}/W{well:03d}.{ext}"
    imagesaver.load()

    for base in imageloader.param_ranges['base']:
        for well in imageloader.param_ranges['well']:
            stitch_keyence(imageloader, imagesaver, dict(base=base, well=well), 'layer')


def stitch_keyence(imageloader, imagesaver, static_params, stack_param, **kwargs):
    print('starting stitching:', static_params)
    images, param_list = imageloader.load_params(stack_param=stack_param, **static_params)
    print(' images loaded', static_params)

    # images, param_list = imageloader.load_params(base=1, well=1, stack_param='layer')
    # images, param_list = imageloader.load_params(base=1, well=1, part=[1,2,3,4, 10,11,12,13, 22,23,24,25, 36,37,38,39], stack_param='layer')
    # parts = [131,132,133,134,135,154,155,156,157,158,178,179,180,181,182,203,204,205,206,207,228,229,230,231,232]
    # images, param_list = imageloader.load_params(base=1, well=1, part=parts, stack_param='layer')

    downscale_factor = 1
    method = 'm2stitch_new'

    for i in range(len(images)):
        images[i] = images[i].max(axis=2).astype('uint16')
        if downscale_factor > 1:
            images[i] = downscale_mean(images[i], downscale_factor).astype('uint16')
        # fig, axis = plt.subplots()
        # axis.imshow(images[i][:,:,0])
        # axis.set_title(param_list[i]['part'])
    # plt.show()
    print('final shape', images[0].shape)

    # images = (illum_correction(images) * 65535).astype('uint16')

    # composite = stitch_grid(images, param_list, 6, x_skew=-30, y_skew=30)
    print(' starting m2stitch:', static_params)
    composite = stitch_grid_m2stitch(images, param_list, imageloader, get_grid_position_keyence, ncc_threshold=0.1,
                                     **kwargs)
    # composite = stitch_grid_metadata(images, param_list, imageloader, get_position_keyence)
    print(' done m2stitch', static_params)
    del images
    image = composite.full_image()
    del composite
    params = param_list[0].copy()
    params['method'] = 'm2stitch'
    params['resolution'] = 'full'
    params['ext'] = 'tif'
    print(' saving image:', imagesaver.to_path(params))
    print(image.min(), image.max(), image.dtype)
    skimage.io.imsave(imagesaver.make_path(params), image, compression='deflate')

    image = downscale_mean(image, 3)
    image = image / np.percentile(image, 99.9, axis=(0, 1)) * 255
    image[image < 0] = 0
    image[image > 255] = 255
    image = image.astype('uint8')
    image[..., 0] //= 2
    image[..., 2] //= 2
    image[..., 0] += image[..., 3] // 2
    image[..., 2] += image[..., 3] // 2
    params['resolution'] = 'rgb'
    params['ext'] = 'png'
    print(' saving image:', imagesaver.to_path(params))
    skimage.io.imsave(imagesaver.make_path(params), image[..., :3])
    print('done with:', static_params)


if __name__ == "__main__":
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    stitch_keyence_all(sys.argv[1])