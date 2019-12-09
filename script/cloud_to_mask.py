#!/usr/bin/env python
# -*- coding: utf-8 -*

""".
Project 3d pointcloud to pixel.
"""

import image_geometry
import rospy
from sensor_msgs.msg import CameraInfo
import sys
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge


class Calc_p2d():
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/cloud_mask", Image, queue_size=10)
        self.input_cloud = rospy.get_param(
            '~input_cloud', "/prosilica_cloud/output")
        self.camera_info = rospy.get_param(
            '~camera_info', "/prosilica/camera_info")
        self.width = 2050
        self.height = 2448
        self.cm = image_geometry.cameramodels.PinholeCameraModel()
        self.load_camera_info()
        self.subscribe()

    def load_camera_info(self):
        ci = rospy.wait_for_message(self.camera_info, CameraInfo)
        self.cm.fromCameraInfo(ci)
        self.P_np = np.array(self.cm.P)
        rospy.loginfo("load camera info")

    def subscribe(self):
        self.image_sub = rospy.Subscriber(
            self.input_cloud, PointCloud2, self.callback)

    def callback(self, msg):
        mask = np.zeros((self.width, self.height), dtype=np.uint8)
        points_np = np.array(list(pc2.read_points(
            msg, skip_nans=True,
            field_names=("x", "y", "z")))).T
        if (points_np.shape[0] != 0):
            points_np = np.pad(points_np, [(0, 1), (0, 0)],
                               'constant',  constant_values=1)
            p2d = np.matmul(self.P_np, points_np)
            p2d = p2d[:2] / p2d[2]
            p2d = p2d.astype(np.int16)
            x_out = np.where(p2d[1] >= self.cm.height)
            y_out = np.where(p2d[0] >= self.cm.width)
            x_in = np.delete(p2d[1], x_out)
            x_in = np.delete(x_in, y_out)
            y_in = np.delete(p2d[0], x_out)
            y_in = np.delete(y_in, y_out)
            index = (x_in, y_in)
            mask[index] = 255

        else:
            rospy.loginfo("nothing in the gripper")
            mask[:] = 255
        msg_out = self.bridge.cv2_to_imgmsg(mask, "mono8")
        msg_out.header.stamp = msg.header.stamp
        self.pub.publish(msg_out)

def main(args):
    rospy.init_node("cloud_to_mask", anonymous=False)
    calc_p2d = Calc_p2d()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
