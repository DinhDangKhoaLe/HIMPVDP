#!/usr/bin/env python
import multiprocessing as mp
import rospy
import numpy
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TwistStamped
from fetchbot_msgs.msg import ArmState



class FetchController(mp.Process):
    def __init__(self):
        # init node
        # super().__init__(name="FetchController")
        rospy.init_node('FetchController', anonymous=True)
        self.arm_controller_pubisher = rospy.Publisher('/arm_controller_twist_test', TwistStamped, queue_size=10)
        # self.arm_gripper_pubisher = rospy.Publisher('/arm_controller_twist_test', TwistStamped, queue_size=10)
        rospy.Subscriber("/arm_joint", ArmState, self.armJointCallback)
        self.rate = rospy.Rate(10) # 10hz
        
        # private members
        self.end_effector_pose = None
        self.predict_pose_trajectory = None
        self.predict_gripper_trajectory = None
        self.predict_time_stamped = None
        self.predict_velocity = None
        self.velocity_scale = 0.1

    def setPredictPoseTrajectory(self, predict_pose_trajectory, predict_gripper_trajectory,predict_time_stamped):
        self.predict_pose_trajectory = predict_pose_trajectory
        self.predict_gripper_trajectory = predict_gripper_trajectory
        self.predict_time_stamped = predict_time_stamped
        rospy.loginfo("END EFFECTOR POSE: " + self.end_effector_pose)
        
    def armJointCallback(self,data):
        euler = euler_from_quaternion(data.orientation)
        self.end_effector_pose = [data.pose.position.x, data.pose.position.y, data.pose.position.z, euler[0], euler[1], euler[2]]

        rospy.loginfo("END EFFECTOR POSE: " + self.end_effector_pose)

    def calculateEndEffectorVelocity(self):
        # calculate end effector velocity
        self.predict_velocity = TwistStamped()
        self.predict_velocity.header.stamp = rospy.Time.now()
        # self.predict_velocity.twist.linear.x = (self.predict_pose_trajectory[0] - self.end_effector_pose[0])
        # self.predict_velocity.twist.linear.y = (self.predict_pose_trajectory[1] - self.end_effector_pose[1])
        # self.predict_velocity.twist.linear.z = (self.predict_pose_trajectory[2] - self.end_effector_pose[2])
        # self.predict_velocity.twist.angular.x = (self.predict_gripper_trajectory[3] - self.end_effector_pose[3])
        # self.predict_velocity.twist.angular.y = (self.predict_gripper_trajectory[4] - self.end_effector_pose[4])
        # self.predict_velocity.twist.angular.z = (self.predict_gripper_trajectory[5] - self.end_effector_pose[5])
        return self.predict_velocity*self.velocity_scale
    
    # main loop in process
    def run(self):
        while not rospy.is_shutdown():
            self.arm_controller_pubisher.publish(self.calculateEndEffectorVelocity())
            self.rate.sleep()
            rospy.loginfo("PUBLISHED TO ARM CONTROLLER " + self.calculateEndEffectorVelocity())
    