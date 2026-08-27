"""Display metric depth images received from the Starling Gazebo model."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image as PilImage
from PIL import ImageTk

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


class DepthViewer(Node):
    """Subscribe to and display a raw depth image in grayscale."""

    def __init__(self) -> None:
        super().__init__("depth_viewer")
        self.declare_parameter("topic", "/starling/raw_depth")
        self.declare_parameter("max_depth", 10.0)

        self.topic = str(self.get_parameter("topic").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        if self.max_depth <= 0.0:
            raise ValueError("max_depth must be positive")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            Image,
            self.topic,
            self.depth_callback,
            qos_profile,
        )

        self.root = tk.Tk()
        self.root.title(f"Depth viewer: {self.topic}")
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.status_label = tk.Label(self.root, text="Waiting for depth...")
        self.status_label.pack()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(10, self.spin_once)

        self.tk_image = None
        self.closed = False
        self.get_logger().info(f"Subscribed to {self.topic}")

    @staticmethod
    def image_to_meters(message: Image) -> np.ndarray:
        """Decode float-meter and uint16-millimeter ROS depth images."""
        encoding = message.encoding.lower()
        if encoding == "32fc1":
            dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
            scale = 1.0
        elif encoding == "64fc1":
            dtype = np.dtype(">f8" if message.is_bigendian else "<f8")
            scale = 1.0
        elif encoding in {"16uc1", "mono16"}:
            dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
            scale = 0.001
        else:
            raise ValueError(f"Unsupported depth encoding: {message.encoding}")

        if message.step % dtype.itemsize != 0:
            raise ValueError("Image step is not aligned to its pixel size")
        row_width = message.step // dtype.itemsize
        if row_width < message.width:
            raise ValueError("Image step is smaller than its width")

        required_bytes = message.height * message.step
        if len(message.data) < required_bytes:
            raise ValueError("Depth image payload is incomplete")

        depth = np.frombuffer(
            message.data,
            dtype=dtype,
            count=message.height * row_width,
        )
        depth = depth.reshape(message.height, row_width)[:, :message.width]
        return depth.astype(np.float32, copy=True) * scale

    def grayscale_depth(self, depth: np.ndarray) -> np.ndarray:
        """Convert raw metric depth to one displayable grayscale channel."""
        display_depth = np.nan_to_num(
            depth,
            nan=self.max_depth,
            posinf=self.max_depth,
            neginf=0.0,
        )
        display_depth = np.clip(display_depth, 0.0, self.max_depth)
        return (display_depth * (255.0 / self.max_depth)).astype(np.uint8)

    def depth_callback(self, message: Image) -> None:
        """Decode and display the latest depth frame."""
        try:
            depth = self.image_to_meters(message)
        except ValueError as error:
            self.get_logger().error(str(error))
            return

        grayscale = self.grayscale_depth(depth)
        self.tk_image = ImageTk.PhotoImage(PilImage.fromarray(grayscale))
        self.image_label.configure(image=self.tk_image)

        valid = depth[np.isfinite(depth) & (depth > 0.0)]
        if valid.size:
            text = (
                f"{message.width}x{message.height} {message.encoding} | "
                f"min {valid.min():.2f} m | max {valid.max():.2f} m"
            )
        else:
            text = (
                f"{message.width}x{message.height} {message.encoding} | "
                "no valid depth"
            )
        self.status_label.configure(text=text)

    def spin_once(self) -> None:
        """Service ROS callbacks without blocking the Tk event loop."""
        if self.closed or not rclpy.ok():
            return
        rclpy.spin_once(self, timeout_sec=0.0)
        self.root.after(10, self.spin_once)

    def close(self) -> None:
        """Stop ROS and close the viewer window."""
        if self.closed:
            return
        self.closed = True
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self.root.destroy()


def main(args=None) -> None:
    """Run the depth-image viewer."""
    rclpy.init(args=args)
    viewer = DepthViewer()
    try:
        viewer.root.mainloop()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
