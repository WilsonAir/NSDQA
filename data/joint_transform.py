import random
import numpy as np
import torchvision.transforms
from PIL import Image, ImageEnhance


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, detections, fp=None, fn=None):
        # assert img.size == mask.size
        for t in self.transforms:
            if not fp is None:
                img, mask, detections, fp, fn = t(img, mask, detections, fp, fn)
            else:
                img, mask, detections = t(img, mask, detections)
        if not fp is None:
            return img, mask, detections, fp, fn
        else:
            return img, mask, detections

# class Resize(object):
#     def __init__(self, size):
#         self.size = tuple(reversed(size))  # size: (h, w)
#
#     def __call__(self, img, mask):
#         assert img.size == mask.size
#         return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.BILINEAR)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        x1 = random.randint(0, w - self.size)
        y1 = random.randint(0, h - self.size)
        return img.crop((x1, y1, x1 + self.size, y1 + self.size)), mask.crop((x1, y1, x1 + self.size, y1 + self.size))


class To_pil(object):
    def __call__(self, img, mask, detections, fp=None, fn=None):
        detections_list = []
        to_pil = torchvision.transforms.ToPILImage()
        for idx_detection in range(detections.shape[0]):
            detections_temp = detections[idx_detection]
            # detections_temp = np.expand_dims(detections_temp,2)
            detections_temp = to_pil(detections_temp)
            detections_list.append(detections_temp)

        if not fp is None:
            return to_pil(img), to_pil(mask), detections_list, to_pil(fp), to_pil(fn)
        return to_pil(img), to_pil(mask), detections_list


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, predictions, fp=None, fn=None):
        if random.random() < 0.5:
            predictions_flip = []
            for item in predictions:
                predictions_flip.append(item.transpose(Image.FLIP_LEFT_RIGHT))
            img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            # img.show()
            # mask.show()
            # predictions_flip[0].show()
            # predictions_flip[1].show()
            # predictions_flip[2].show()
            if not fp is None:
                return img, mask, predictions_flip, fp.transpose(Image.FLIP_LEFT_RIGHT), fn.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask, predictions_flip

        if not fp is None:
            return img, mask, predictions, fp, fn
        return img, mask, predictions


class RandomVertFlip(object):
    def __call__(self, img, mask, predictions, fp=None, fn=None):
        if random.random() < 0.5:
            predictions_flip = []
            for item in predictions:
                predictions_flip.append(item.transpose(Image.FLIP_TOP_BOTTOM))
            img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
            # img.show()
            # mask.show()
            # predictions_flip[0].show()
            # predictions_flip[1].show()
            # predictions_flip[2].show()
            if not fp is None:
                return img, mask, predictions_flip, fp.transpose(Image.FLIP_TOP_BOTTOM), fn.transpose(Image.FLIP_TOP_BOTTOM)
            return img, mask, predictions_flip

        if not fp is None:
            return img, mask, predictions, fp, fn
        return img, mask, predictions


class Enhance(object):
    def __call__(self, img, mask, predictions, fp=None, fn=None):
        # is_gray = img.ndim == 2 or img.shape[1] == 1
        # if is_gray:
        #     img = img.convert('RGB')
        a = random.random()

        if a<1/5:
            enh_col = ImageEnhance.Color(img)
            image_enhance = enh_col.enhance(1.5)
        elif a<1/5:
            enh_bri = ImageEnhance.Brightness(img)
            image_enhance = enh_bri.enhance(1.5)
        elif a<3/5:
            enh_con = ImageEnhance.Contrast(img)
            image_enhance = enh_con.enhance(1.2)
        elif a<4/5:
            enh_sha = ImageEnhance.Sharpness(img)
            image_enhance = enh_sha.enhance(1.5)
        else:
            image_enhance = img

        img = image_enhance
        if not fp is None:
            return img, mask, predictions, fp, fn
        return img, mask, predictions


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, predictions, fp=None, fn=None):
        predictions_flip = []
        for item in predictions:
            predictions_flip.append(item.resize(self.size, Image.BILINEAR))
        if not fp is None:
            return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.BILINEAR), predictions_flip, \
                   fp.resize(self.size, Image.BILINEAR), fn.resize(self.size, Image.BILINEAR)
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.BILINEAR), predictions_flip


class To_tensor(object):
    def __call__(self, img, mask, predictions, fp=None, fn=None):
        to_tensor = torchvision.transforms.ToTensor()
        predictions_flip = []
        for item in predictions:
            predictions_flip.append(to_tensor(item))
        if not fp is None:
            return to_tensor(img), to_tensor(mask), predictions_flip, to_tensor(fp), to_tensor(fn)
        return to_tensor(img), to_tensor(mask), predictions_flip
