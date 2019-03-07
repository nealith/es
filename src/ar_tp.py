#!/usr/bin/env python
"""This script shows an example of using the PyWavefront module."""
import ctypes
import sys
import os
import pyglet
from pyglet.gl import *

glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

from pywavefront import visualization
import pywavefront

import cv2
import numpy as np

from PIL import Image

import csv
import math

MIN_MATCHES = 10


# use to pass image from cv2 to pyglet
def cv2glet(img):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    rows, cols, channels = img.shape
    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols,
                                   height=rows,
                                   format='BGR',
                                   data=raw_img,
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg

lightfv = ctypes.c_float * 4

# Our program class based on pyglet window class
class Window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        capture = kwargs['capture']
        try:
            del kwargs['capture']
        except KeyError:
            pass
        super(Window, self).__init__(*args, **kwargs)
        self.batch = pyglet.graphics.Batch()

        self.capture = capture

        #self.width = 0
        #self.height = 0
        #self.meshes = pywavefront.Wavefront('14079_WWII_Tank_UK_Cromwell_v1_L2.obj')

        #if self.capture.isOpened():
            # get capture property
        #    self.width = int(self.capture.get(3))   # float
        #    self.height = int(self.capture.get(4)) # float
        #    print ("size : ",self.width," ",self.height)

        self.data = []

        # matrix of camera parameters (made up but works quite well for me)
        self.camera_parameters = np.array([[800, 0, self.width/2.0], [0, 800, self.height/2.0], [0, 0, 1]])

        orb = cv2.ORB_create()

        dir_name = os.getcwd()
        with open('../data.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:

                tmp_data = {'reference_kp':None,'reference_des':None,'reference':cv2.imread(os.path.join(dir_name, row['reference']), 0),'reference_name':row['reference'],'model':pywavefront.Wavefront(row['model']),'model_name':row['model']}
                kp, des = orb.detectAndCompute(tmp_data['reference'],None)
                tmp_data['reference_des'] = des
                tmp_data['reference_kp'] = kp
                self.data.append(tmp_data)

        self.sprite = None

        self.rvec = []
        self.tvec = []
        self.model = None

    def on_draw(self):
        #self.get_capture()

        self.clear()
        #glTranslated(0.0, 0.0, -10.0)

        if self.sprite != None:
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width, 0, self.height, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            self.batch.draw()

        if self.model != None:

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            if self.width > 0 and self.height > 0:
                gluPerspective(800, float(self.width)/self.height, 1., 100.)
            glMatrixMode(GL_MODELVIEW)

            glEnable(GL_TEXTURE_2D)
            #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            #glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
            #this one is necessary with texture2d for some reason
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

            global lightfv
            glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
            glEnable(GL_LIGHT0)

            glEnable(GL_LIGHTING)

            if len(self.tvec):
                glTranslated(self.tvec[0]/100, self.tvec[1]/100, -self.tvec[2]/100)

            if len(self.rvec):
                glRotatef(self.rvec[0]*100, 0.0, 1.0, 0.0)
                glRotatef(self.rvec[1]*100, 1.0, 0.0, 0.0)
                glRotatef(self.rvec[2]*100, 0.0, 0.0, 1.0)

            global visualization
            visualization.draw(self.model)

    def update_capture(self):
        ret, frame = self.capture.read()
        self.sprite = pyglet.sprite.Sprite(cv2glet(frame), batch=self.batch)

    def update_ar(self):

        #self.model = None
        ret, frame = self.capture.read()

        homography = None

        # create ORB keypoint detector
        orb = cv2.ORB_create()
        # create BFMatcher object based on hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # load the reference surface that will be searched in the video stream


        suitor = None
        suitor_matches  = []

        kp_frame, des_frame = orb.detectAndCompute(frame, None)

        for data in self.data:
            # match frame descriptors with model descriptors
            matches = bf.match(data['reference_des'], des_frame)
            # sort them in the order of their distance
            # the lower the distance, the better the match
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > MIN_MATCHES and len(matches) > len(suitor_matches):
                suitor = data
                suitor_matches = matches

        if suitor != None:
            matches = suitor_matches

            # differenciate between source points and destination points

            try:

                src_pts = np.float32([data['reference_kp'][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                # compute Homography
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

                v,homography_inv = cv2.invert(homography)

                test_src_pts = cv2.perspectiveTransform(dst_pts, homography_inv)
                print("test",test_src_pts)
                print("src",src_pts)
  



                #if args.rectangle:
                    # Draw a rectangle that marks the found model in the frame
                h, w = suitor['reference'].shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    # project corners into frame
                dst = cv2.perspectiveTransform(pts, homography)

                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)


                self.model = suitor['model']



                Dpts= np.float32([[0,0,0],[0,h-1,0],[w-1,h-1,0],[w-1,0,0]]);

                retval, self.rvec, self.tvec = cv2.solvePnP(Dpts,dst,self.camera_parameters,None)

                        #print(suitor['reference_name'])
                        #print(self.rvec,'--', self.tvec)

            except:
                pass



        else:
            print ("Not enough matches found")

        self.sprite = pyglet.sprite.Sprite(cv2glet(frame), batch=self.batch)

tdt = 0

window = None
rotation = 0



def update(dt):
    global window
    global tdt
    tdt += dt
    #window.update_capture()

    #if tdt*1000.0 > 1000.0/60.0:
    window.update_ar()
    tdt = 0




def main():
    global window
    capture = cv2.VideoCapture(0)
    if capture.isOpened():
        # get capture property
        width = int(capture.get(3))   # float
        height = int(capture.get(4)) # float
        print ("size : ",width," ",height)


        window = Window(capture=capture,width=width, height=height, caption='Pyglet')
        pyglet.clock.schedule(update)
        pyglet.app.run()


if __name__ == '__main__':
    main()
