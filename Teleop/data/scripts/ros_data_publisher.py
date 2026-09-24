import rospy
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
from threading import Thread, Lock


class RobotDataPublisher:
    def __init__(
        self,
        rate_hz=25,
        init_node: bool = True,
        joint_topic: str = "/robot2/joint_states",
        rgb_image_topic: str = "/camera_image",
    ):
        if init_node:
            rospy.init_node("robot_data_publisher")

        # Publishers
        self.joint_pub = rospy.Publisher(joint_topic, JointState, queue_size=10)

        self.image_pub = rospy.Publisher(rgb_image_topic, Image, queue_size=10)
        self.bridge = CvBridge()

        # Data and locks
        self.joint_angles = []
        self.joint_names = []
        self.image = None
        self.lock = Lock()

        # Threading and rate
        self.rate_hz = rate_hz
        self.thread = None
        self.running = False

    def update_data(
        self, joint_angles, joint_names, image
    ):
        """Update the data to be published."""
        with self.lock:
            self.joint_angles = joint_angles
            self.joint_names = joint_names
            self.image = image

    def publish_joint_angles(self):
        """Publish joint angles as JointState message."""
        with self.lock:
            if len(self.joint_angles) and len(self.joint_names):
                joint_state_msg = JointState()
                joint_state_msg.header.stamp = rospy.Time.now()
                joint_state_msg.name = self.joint_names
                joint_state_msg.position = self.joint_angles
                self.joint_pub.publish(joint_state_msg)
            else:
                return

    def publish_image(self):
        """Publish image as ROS Image message."""
        with self.lock:
            if self.image is None:
                return
            ros_image = self.bridge.cv2_to_imgmsg(self.image, encoding="rgb8")
        self.image_pub.publish(ros_image)

    def start(self):
        """Start the publishing thread."""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = Thread(target=self.run)
            self.thread.start()

    def stop(self):
        """Stop the publishing thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def run(self):
        """Main loop to publish data periodically."""
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and self.running:
            self.publish_joint_angles()
            self.publish_image()
            rate.sleep()
