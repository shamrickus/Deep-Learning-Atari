import numpy
from skimage.transform import rescale


class BaseROM:
    name = None

    def process_image(self, image):
        return self.rescale(self.gray_scale(self.crop(image)))

    def rescale(self, image):
        result_size = 84
        assert image.shape[0] == image.shape[1]
        return rescale(image, result_size / image.shape[0])

    def crop(self, image):
        return image

    def gray_scale(self, image):
        return numpy.dot(image[..., :3], [0.299, 0.587, 0.114])

    def actions(self):
        return range(0, 17)


class Breakout(BaseROM):
    def __init__(self):
        self.name = "Breakout.a26"

    def crop(self, image):
        return image[35:-15, ...]

    def actions(self):
        return range(0, 5)


class Beamrider(BaseROM):
    def __init__(self):
        self.name = "Beamrider.a26"


class Enduro(BaseROM):
    def __init__(self):
        self.name = "Enduro.a26"


class Pong(BaseROM):
    def __init__(self):
        self.name = "Pong.a26"


class Qbert(BaseROM):
    def __init__(self):
        self.name = "Qbert.a26"


class Seaquest(BaseROM):
    def __init__(self):
        self.name = "Seaquest.a26"


class SpaceInvaders(BaseROM):
    def __init__(self):
        self.name = "SpaceInvaders.a26"
