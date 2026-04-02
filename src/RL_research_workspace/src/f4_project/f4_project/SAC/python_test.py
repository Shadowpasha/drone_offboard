import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor

def node_spin():
    global executor
    executor.spin()

rclpy.init()
node = rclpy.create_node("training")
executor = MultiThreadedExecutor()
executor.add_node(node)

et = threading.Thread(target=node_spin)
et.start()
