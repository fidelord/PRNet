import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time
from skimage.transform import warp, resize, estimate_transform
from PRNet.predictor import PosPrediction

class PRN:
    ''' Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    Args:
        is_dlib(bool, optional): If true, dlib is used for detecting faces.
        prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
    '''
    def __init__(self, is_dlib = False, prefix = '.'):

        # resolution of input and output image size.
        self.resolution_inp = 256
        self.resolution_op = 256

        #---- load detectors
        #if is_dlib:
            #import dlib
            #detector_path = os.path.join(prefix, 'Data/net-data/mmod_human_face_detector.dat')
          #  self.face_detector = dlib.cnn_face_detection_model_v1(
                    #detector_path)

        #---- load PRN 
        self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
        prn_path = os.path.join(prefix, 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.data-00000-of-00001'):
            print("please download PRN trained model first.")
            exit()
        #self.pos_predictor.restore(prn_path)

        # uv file
        self.uv_kpt_ind = np.loadtxt(prefix + '/Data/uv-data/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt
        self.face_ind = np.loadtxt(prefix + '/Data/uv-data/face_ind.txt').astype(np.int32) # get valid vertices in the pos map
        self.triangles = np.loadtxt(prefix + '/Data/uv-data/triangles.txt').astype(np.int32) # ntri x 3
        
        self.uv_coords = self.generate_uv_coords()        

    def generate_uv_coords(self):
        resolution = self.resolution_op
        uv_coords = np.meshgrid(range(resolution),range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1,2,0])
        uv_coords = np.reshape(uv_coords, [resolution**2, -1]);
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0], 1])))
        return uv_coords

    #def dlib_detect(self, image):
        #return self.face_detector(image, 1)

    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (256,256,3) array. value range: 0~1
        Returns:
            pos: the 3D position map. (256, 256, 3) array.
        '''
        return self.pos_predictor(image)

    def process(self, input, image_info=True):
              """
              process image with crop operation.
              Args:
                  input: (h,w,3) array or str(image path). image value range: 1~255.
                  image_info (optional): if not None, indicates that the input image is already
                      the cropped face (bounding box == full image). In that case, we skip the
                      usual similarity transform and just resize to resolution_inp.
              Returns:
                  pos: the 3D position map. (resolution_op, resolution_op, 3).
              """
              if isinstance(input, str):
                  try:
                      image = imread(input)
                  except IOError:
                      print("error opening file:", input)
                      return None
              else:
                  image = input

              if image.ndim < 3:
                  image = np.tile(image[:, :, np.newaxis], [1, 1, 3])

              if True:
                  # Input is exactly the bounding‐box crop → no need to compute src_pts or warp
                  # Just normalize and resize straight to resolution_inp
                  image_norm = image.astype(np.float32) / 255.0
                  cropped_image = resize(
                      image_norm,
                      (self.resolution_inp, self.resolution_inp),
                      mode='reflect',
                      anti_aliasing=True
                  )
              else:
                  # Original behavior: detect face via dlib and perform similarity‐based crop
                  detected_faces = self.dlib_detect(image)
                  if len(detected_faces) == 0:
                      print('warning: no detected face')
                      return None

                  d = detected_faces[0].rect  # assume one face
                  left, right = d.left(), d.right()
                  top, bottom = d.top(), d.bottom()
                  old_size = (right - left + bottom - top) / 2
                  center = np.array([
                      right - (right - left) / 2.0,
                      bottom - (bottom - top) / 2.0 + old_size * 0.14
                  ])
                  size = int(old_size * 1.58)

                  # define source and destination points for similarity transform
                  src_pts = np.array([
                      [center[0] - size / 2, center[1] - size / 2],
                      [center[0] - size / 2, center[1] + size / 2],
                      [center[0] + size / 2, center[1] - size / 2]
                  ])
                  DST_PTS = np.array([
                      [0, 0],
                      [0, self.resolution_inp - 1],
                      [self.resolution_inp - 1, 0]
                  ])
                  tform = estimate_transform('similarity', src_pts, DST_PTS)

                  image_norm = image.astype(np.float32) / 255.0
                  cropped_image = warp(
                      image_norm,
                      tform.inverse,
                      output_shape=(self.resolution_inp, self.resolution_inp)
                  )

              # run the network
              cropped_pos = self.net_forward(cropped_image)

              # restore to original coordinate space (only needed if image_info was None)
              if image_info is None:
                  cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
                  z = cropped_vertices[2, :].copy() / tform.params[0, 0]
                  cropped_vertices[2, :] = 1
                  vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
                  vertices = np.vstack((vertices[:2, :], z))
                  pos = np.reshape(vertices.T, [self.resolution_op, self.resolution_op, 3])
              else:
                  # If we skipped cropping, just upsample/downsample network output
                  pos = resize(
                      cropped_pos,
                      (self.resolution_op, self.resolution_op, 3),
                      mode='reflect',
                      anti_aliasing=True
                  ).astype(np.float32)

              return pos
            
    def get_landmarks(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        return kpt


    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1])
        vertices = all_vertices[self.face_ind, :]

        return vertices

    def get_colors_from_texture(self, texture):
        '''
        Args:
            texture: the texture map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        all_colors = np.reshape(texture, [self.resolution_op**2, -1])
        colors = all_colors[self.face_ind, :]

        return colors


    def get_colors(self, image, vertices):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] # n x 3

        return colors








