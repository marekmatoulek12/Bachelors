import rclpy
from rclpy.node import Node

class HelloWorld(Node):
    def __init__(self, name: str):
        super().__init__(name)

        self.create_timer(1, callback=self.callback)

    def callback(self):
        self.get_logger().info("Hello world!")
        pass


def main(args=None):
    rclpy.init(args=args)

    node = HelloWorld("hello_world_node")
    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()