import time
import cv2
import numpy as np

class ImgGreyscale():

    def run(self, img_arr):
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        return img_arr

    def shutdown(self):
        pass

class ImgWriter():

    def __init__(self, filename):
        self.filename = filename

    def run(self, img_arr):
        cv2.imwrite(self.filename, img_arr)

    def shutdown(self):
        pass

class ImgBGR2RGB():

    def run(self, img_arr):
        if img_arr is None:
            return None
        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            return img_arr
        except:
            return None

    def shutdown(self):
        pass

class ImgRGB2BGR():

    def run(self, img_arr):
        if img_arr is None:
            return None
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        return img_arr

    def shutdown(self):
        pass

class ImageScale():

    def __init__(self, scale):
        self.scale = scale

    def run(self, img_arr):
        if img_arr is None:
            return None
        try:
            return cv2.resize(img_arr, (0,0), fx=self.scale, fy=self.scale)
        except:
            return None

    def shutdown(self):
        pass

class ImageRotateBound():
    '''
    credit:
    https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    '''

    def __init__(self, rot_deg):
        self.rot_deg = rot_deg

    def run(self, image):
        if image is None:
            return None

        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
    
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -self.rot_deg, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
    
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
    
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
    
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def shutdown(self):
        pass

class ImgCanny():

    def __init__(self, low_threshold=60, high_threshold=110):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        
    def run(self, img_arr):
        return cv2.Canny(img_arr, 
                         self.low_threshold, 
                         self.high_threshold)

    def shutdown(self):
        pass
    

class ImgGaussianBlur():

    def __init__(self, kernal_size=5):
        self.kernal_size = kernal_size
        
    def run(self, img_arr):
        return cv2.GaussianBlur(img_arr, 
                                (self.kernel_size, self.kernel_size), 0)

    def shutdown(self):
        pass


def region_of_interest(shape=(120, 160, 3), roi_region=None):
    '''
    Create a function to apply a region of interest mask to an image.
    The pixels that fall within the region are conserved,
    the pixels outside the region are cleared.

    parameters:
        shape: shape of image's numpy array (height, width, depth)
        roi_region: array of closed polygons, like [[(0, 95), (0, 120), (160, 120), (160, 95), (80, 45), (40, 45)]]
                    Each polygon is an array of 2D integer vertices.
                    Each polygon is closed by automatically connecting the last vertex  to the first vertex.
    returns:
        A lambda function that applies region of interest mask to an image and return masked image
    '''
    if roi_region is None:
        return lambda x: x      # identity function if no region

    # function to apply mask to clear region
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, np.array(roi_region), 255)
    return lambda x: cv2.bitwise_and(x, mask)


class ImgRegionOfInterest:
    '''
    A part that conserves pixels in the given region and clears pixels outside of region.
    The region is specified as an array of polygons.  Each polygon is an array of vertices;
    the polygon is always closed by connecting the last vertex with the first vertex.
    '''

    def __init__(self, shape, region):
        '''
            Create a mask with region and apply to img_arr
            parameters:
                shape: shape of images to be masked (height, width, depth)
                region: array of closed polygons, like [[(0, 95), (0, 120), (160, 120), (160, 95), (80, 45), (40, 45)]]
                        Each polygon is an array of 2D integer vertices.
                        Each polygon is closed by automatically connecting the last vertex  to the first vertex.
        '''
        self.region_mask = region_of_interest(shape, region)

    def run(self, img_arr):
        '''
            Apply region mask to img_arr, save back to img_arr
            NOTE that this caches the region mask and so presumes
                 that all images are the same dimensions

            parameters:
                img_arr: numpy array that represents an image
            returns:
                numpy array representing the masked image.
                If there is no region, the original image is returned.
        '''
        if img_arr is None:
            return None
        try:
            if self.region_mask is not None:
                #
                # mask the image
                #
                masked_image = self.region_mask(img_arr)

            else:
                #
                # no region, just return the original image
                #
                masked_image = img_arr

                # plt.imshow(masked_image)
                # plt.show() # show in a window (until window is closed)

            return masked_image
        except:
            return img_arr

    def shutdown(self):
        pass


class ArrowKeyboardControls:
    '''
    kind of sucky control, only one press active at a time. 
    good enough for a little testing.
    requires that you have an CvImageView open and it has focus.
    '''
    def __init__(self):
        self.left = 2424832
        self.right = 2555904
        self.up = 2490368
        self.down = 2621440
        self.codes = [self.left, self.right, self.down, self.up]
        self.vec = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def run(self):
        code = cv2.waitKeyEx(delay=100)
        for iCode, keyCode in enumerate(self.codes):
            if keyCode == code:
                return self.vec[iCode]
        return (0., 0.)
        
        
        
class Pipeline():
    def __init__(self, steps):
        self.steps = steps
    
    def run(self, val):
        for step in self.steps:
            f = step['f']
            args = step['args']
            kwargs = step['kwargs']
            
            val = f(val, *args, **kwargs)
        return val
    
class CvCam(object):
    def __init__(self, image_w=160, image_h=120, image_d=3, iCam=0):

        self.frame = None
        self.cap = cv2.VideoCapture(iCam)
        self.running = True
        self.cap.set(3, image_w)
        self.cap.set(4, image_h)

    def poll(self):
        if self.cap.isOpened():
            ret, self.frame = self.cap.read()

    def update(self):
        '''
        poll the camera for a frame
        '''
        while(self.running):
            self.poll()

    def run_threaded(self):
        return self.frame

    def run(self):
        self.poll()
        return self.frame

    def shutdown(self):
        self.running = False
        time.sleep(0.2)
        self.cap.release()


class CvImageView(object):

    def run(self, image):
        if image is None:
            return
        try:
            cv2.imshow('frame', image)
            cv2.waitKey(1)
        except:
            pass

    def shutdown(self):
        cv2.destroyAllWindows()
