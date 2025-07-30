# src/final_challenge_pkg/robot_controller.py
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import os

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.model(x)

class RobotAIIntegrationNode(Node):
    def __init__(self):
        super().__init__('robot_ai_integration_node')
        self.get_logger().info('Robot AI Integration Node has been started.')

        self.mnist_model = MLP()
        model_path = os.path.expanduser("~/physicalai-prac2/final_challenge_robot_ai/model/mnist_mlp_model.pt")
        try:
            if os.path.exists(model_path):
                self.mnist_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.mnist_model.eval()
                self.get_logger().info(f'MNIST MLP model loaded successfully from: {model_path}')
            else:
                self.get_logger().warn(f'MNIST MLP model not found at: {model_path}. Please ensure train_mlp_mnist.py was run to generate it.')
        except Exception as e:
            self.get_logger().error(f'Failed to load MNIST MLP model: {e}')

        # AI推論のダミー部分：2秒ごとに推論を実行
        self.inference_timer = self.create_timer(2.0, self.inference_callback)
        self.get_logger().info('AI inference will be simulated every 2 seconds.')

    def inference_callback(self):
        if hasattr(self, 'mnist_model'):
            dummy_input = torch.randn(1, 1, 28, 28) # ランダムな画像を生成
            try:
                with torch.no_grad():
                    output = self.mnist_model(dummy_input)
                    _, predicted = torch.max(output, 1)
                    self.get_logger().info(f'Simulating AI inference: Predicted digit: {predicted.item()}')
            except Exception as e:
                self.get_logger().error(f'Error during dummy AI inference: {e}')
        else:
            self.get_logger().warn('AI model not loaded, cannot perform inference.')

def main(args=None):
    rclpy.init(args=args)
    robot_node = RobotAIIntegrationNode()
    rclpy.spin(robot_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
