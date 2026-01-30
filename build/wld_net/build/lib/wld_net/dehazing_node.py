import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the package directory to sys.path to ensure modules can be imported
# This assumes the node is running from within the installed package
# or the source directory structure is maintained.
# In a standard install, `wld_net` imports should work if `setup.py` is correct.
# However, WLD-Net's internal imports (like `import Feature_Processing`) might need help
# if they are not relative or absolute packaged imports.
# Based on `test.py`, they are local imports.
# We might need to adjust imports in the original files or add the directory to sys.path.
# For now, let's try to add the directory of this file to sys.path.

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

try:
    from . import dehazing_model
    from . import Feature_Processing
except ImportError:
    # Fallback for direct execution or if installed differently
    import dehazing_model
    import Feature_Processing

class DehazingNode(Node):
    def __init__(self):
        super().__init__('dehazing_node')

        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/camera/image_dehazed')
        # Default input size (can be resized, but model might expect specific dims or multiples)
        # The original code uses (1024, 1024) or (512, 512).
        # We'll use the incoming image size, but note that the model architecture (DWT)
        # usually requires dimensions divisible by some factor (e.g., 2^n).
        # WLD-Net uses DWT with sampling=2, and multiple levels.
        # It's safer to ensure dimensions are divisible by 16 or 32.
        
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        if not self.model_path:
            # Default to a model in the models directory relative to this script
            # Assuming models are installed or present in src
            # In a ROS 2 install, non-python files need to be specified in data_files.
            # For now, let's try to find it in the source dir if not provided.
            possible_model_path = current_dir.parent / 'models' / 'RD_dehazing_model_final.pth'
            if possible_model_path.exists():
                self.model_path = str(possible_model_path)
            else:
               self.get_logger().warn("No model_path provided and could not find default model. Please provide model_path parameter.")

        self.get_logger().info(f"Loading model from: {self.model_path}")

        # Model Init
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Running on device: {self.device}")

        self.model = dehazing_model.Dehazing_Model()
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            # We don't exit, just won't process correctly
        
        self.model.to(self.device)
        self.model.eval()

        self.bridge = CvBridge()

        # Subscribers and Publishers
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Image, self.output_topic, 10)
        
        self.get_logger().info(f"Subscribed to {self.input_topic}")
        self.get_logger().info(f"Publishing to {self.output_topic}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Preprocess
        # Original code:
        # data_mp=transform(image)
        # data_mp=data_mp.unsqueeze(0)
        # transform is transforms.Compose([transforms.ToTensor()]) which converts [0, 255] to [0.0, 1.0]
        # and HWC to CHW.
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to Tensor, CHW, float 0-1
        tensor_img = torch.from_numpy(rgb_image.transpose((2, 0, 1))).float().div(255.0)
        tensor_img = tensor_img.unsqueeze(0).to(self.device)

        # Handle size padding if necessary.
        # For simplicity, we assume generic sizes work or user provides correct size.
        # But DWT usually breaks if not divisible by 2^depth.
        # Let's pad to multiple of 16 just in case.
        h, w = tensor_img.shape[2], tensor_img.shape[3]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        
        if pad_h > 0 or pad_w > 0:
            tensor_img = torch.nn.functional.pad(tensor_img, (0, pad_w, 0, pad_h), mode='reflect')
            # self.get_logger().info(f"Padded image from {h}x{w} to {h+pad_h}x{w+pad_w}")

        # Normalize (WLD-Net uses specific normalize function)
        tensor_img = Feature_Processing.normalize(tensor_img)

        # Inference
        try:
            with torch.no_grad():
                output = self.model(tensor_img)
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            return
            
        # Postprocess
        output = Feature_Processing.denormalize(output)
        
        # Crop padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]

        output = output.squeeze(0).cpu().numpy().transpose((1, 2, 0)) # HWC
        output = np.clip(output, 0, 1) * 255.0
        output = output.astype(np.uint8)
        
        # Convert RGB back to BGR for OpenCV/ROS
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(output_bgr, "bgr8")
            out_msg.header = msg.header # Preserve timestamp and frame_id
            self.publisher.publish(out_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error (Publish): {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DehazingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
