/**
 * @file carla_bridge_node.hpp
 * @brief High-performance ROS 2 bridge for CARLA DRL pipeline
 * 
 * This node bridges communication between CARLA simulator (Python 3.6) and
 * DRL agents (Python 3.12) using ZeroMQ for low-latency IPC and ROS 2
 * for standardized robotics messaging.
 */

#ifndef CARLA_BRIDGE_NODE_HPP
#define CARLA_BRIDGE_NODE_HPP

#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <queue>
#include <mutex>

// ROS 2 includes
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>

// ZeroMQ includes
#include <zmq.hpp>

// Standard includes
#include <nlohmann/json.hpp>

namespace carla_ros2_bridge {

/**
 * @brief Configuration for CARLA bridge node
 */
struct BridgeConfig {
    // ZeroMQ settings
    std::string zmq_pub_address = "tcp://localhost:5555";
    std::string zmq_sub_address = "tcp://localhost:5556";
    int zmq_timeout_ms = 1000;
    
    // ROS 2 settings
    std::string node_name = "carla_bridge";
    std::string namespace_name = "/carla";
    
    // Topic names
    std::string camera_image_topic = "/carla/camera/image";
    std::string camera_depth_topic = "/carla/camera/depth";
    std::string vehicle_state_topic = "/carla/vehicle/state";
    std::string vehicle_pose_topic = "/carla/vehicle/pose";
    std::string environment_reward_topic = "/carla/environment/reward";
    std::string environment_done_topic = "/carla/environment/done";
    std::string environment_info_topic = "/carla/environment/info";
    std::string vehicle_control_topic = "/carla/vehicle/control";
    std::string environment_reset_topic = "/carla/environment/reset";
    
    // Performance settings
    size_t publisher_queue_size = 10;
    size_t subscriber_queue_size = 10;
    double publish_rate_hz = 30.0;
    bool use_sim_time = true;
    
    // Image processing
    bool compress_images = true;
    std::string image_encoding = "bgr8";
    int jpeg_quality = 80;
};

/**
 * @brief Main CARLA ROS 2 bridge node
 */
class CarlaBridgeNode : public rclcpp::Node {
public:
    /**
     * @brief Constructor
     * @param config Node configuration
     */
    explicit CarlaBridgeNode(const BridgeConfig& config = BridgeConfig{});
    
    /**
     * @brief Destructor
     */
    ~CarlaBridgeNode();
    
    /**
     * @brief Initialize the bridge node
     * @return True if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Start the bridge operation
     */
    void start();
    
    /**
     * @brief Stop the bridge operation
     */
    void stop();

private:
    // Configuration
    BridgeConfig config_;
    
    // ROS 2 publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_depth_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr vehicle_state_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr vehicle_pose_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr environment_reward_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr environment_done_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr environment_info_pub_;
    
    // ROS 2 subscribers
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vehicle_control_sub_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr environment_reset_sub_;
    
    // Transform broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // ZeroMQ components
    std::unique_ptr<zmq::context_t> zmq_context_;
    std::unique_ptr<zmq::socket_t> zmq_subscriber_;
    std::unique_ptr<zmq::socket_t> zmq_publisher_;
    
    // Threading
    std::atomic<bool> running_;
    std::thread zmq_receiver_thread_;
    std::thread ros_publisher_thread_;
    
    // Message queues
    std::queue<nlohmann::json> sensor_data_queue_;
    std::queue<geometry_msgs::msg::Twist> control_command_queue_;
    std::mutex sensor_data_mutex_;
    std::mutex control_command_mutex_;
    
    // Performance monitoring
    std::chrono::steady_clock::time_point last_stats_time_;
    size_t messages_received_;
    size_t messages_published_;
    
    /**
     * @brief Initialize ZeroMQ communication
     * @return True if initialization successful
     */
    bool initializeZMQ();
    
    /**
     * @brief Initialize ROS 2 publishers and subscribers
     * @return True if initialization successful
     */
    bool initializeROS2();
    
    /**
     * @brief ZeroMQ receiver thread function
     */
    void zmqReceiverLoop();
    
    /**
     * @brief ROS 2 publisher thread function
     */
    void rosPublisherLoop();
    
    /**
     * @brief Process received sensor data from CARLA
     * @param sensor_data JSON sensor data
     */
    void processSensorData(const nlohmann::json& sensor_data);
    
    /**
     * @brief Vehicle control command callback
     * @param msg Control command message
     */
    void vehicleControlCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    
    /**
     * @brief Environment reset callback
     * @param msg Reset message
     */
    void environmentResetCallback(const std_msgs::msg::Empty::SharedPtr msg);
    
    /**
     * @brief Send control command to CARLA via ZeroMQ
     * @param control_msg Control command
     */
    void sendControlCommand(const geometry_msgs::msg::Twist& control_msg);
    
    /**
     * @brief Send reset signal to CARLA via ZeroMQ
     */
    void sendResetSignal();
    
    /**
     * @brief Update performance statistics
     */
    void updateStatistics();
    
    /**
     * @brief Log performance statistics
     */
    void logStatistics();
};

} // namespace carla_ros2_bridge

#endif // CARLA_BRIDGE_NODE_HPP
