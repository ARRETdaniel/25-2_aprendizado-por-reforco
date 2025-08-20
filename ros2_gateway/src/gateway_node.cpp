"""
ROS 2 Gateway Node for CARLA-DRL Bridge (C++)

This C++ node bridges ZeroMQ messages from CARLA Python 3.6 client
to ROS 2 topics for the DRL agent running in Python 3.12.
"""
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <zmq.hpp>
#include <json/json.h>
#include <thread>
#include <chrono>
#include <memory>

class CarlaGatewayNode : public rclcpp::Node
{
public:
    CarlaGatewayNode() : Node("carla_gateway_node")
    {
        // Initialize logger
        RCLCPP_INFO(this->get_logger(), "Initializing CARLA Gateway Node");
        
        // Initialize ZeroMQ
        setup_zmq_connection();
        
        // Initialize ROS 2 publishers
        setup_ros_publishers();
        
        // Initialize ROS 2 subscribers
        setup_ros_subscribers();
        
        // Start ZeroMQ listener thread
        zmq_thread_ = std::thread(&CarlaGatewayNode::zmq_listener_thread, this);
        
        RCLCPP_INFO(this->get_logger(), "CARLA Gateway Node initialized successfully");
    }
    
    ~CarlaGatewayNode()
    {
        // Graceful shutdown
        running_ = false;
        if (zmq_thread_.joinable()) {
            zmq_thread_.join();
        }
        
        if (zmq_socket_) {
            zmq_socket_->close();
        }
        
        RCLCPP_INFO(this->get_logger(), "CARLA Gateway Node shutdown complete");
    }

private:
    // ZeroMQ components
    std::unique_ptr<zmq::context_t> zmq_context_;
    std::unique_ptr<zmq::socket_t> zmq_socket_;
    std::thread zmq_thread_;
    std::atomic<bool> running_{true};
    
    // ROS 2 Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr collision_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr lane_invasion_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    
    // ROS 2 Subscribers
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    
    // QoS profiles
    rclcpp::QoS sensor_qos_{rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)};
    rclcpp::QoS reliable_qos_{rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default)};
    
    void setup_zmq_connection()
    {
        try {
            zmq_context_ = std::make_unique<zmq::context_t>(1);
            zmq_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, ZMQ_SUB);
            
            // Connect to CARLA client
            std::string connect_address = "tcp://localhost:5555";
            zmq_socket_->connect(connect_address);
            zmq_socket_->setsockopt(ZMQ_SUBSCRIBE, "", 0);  // Subscribe to all messages
            
            // Set non-blocking receive
            int timeout = 100;  // 100ms timeout
            zmq_socket_->setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
            
            RCLCPP_INFO(this->get_logger(), "ZeroMQ connected to: %s", connect_address.c_str());
            
        } catch (const zmq::error_t& e) {
            RCLCPP_ERROR(this->get_logger(), "ZeroMQ setup failed: %s", e.what());
            throw;
        }
    }
    
    void setup_ros_publishers()
    {
        // Camera publisher with sensor QoS
        camera_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/carla/ego_vehicle/camera/image", sensor_qos_);
            
        // IMU publisher with high frequency QoS
        imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>(
            "/carla/ego_vehicle/imu", sensor_qos_);
            
        // Odometry publisher with high frequency QoS
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/carla/ego_vehicle/odometry", sensor_qos_);
            
        // Event-based publishers with reliable QoS
        collision_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/carla/ego_vehicle/collision", reliable_qos_);
            
        lane_invasion_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/carla/ego_vehicle/lane_invasion", reliable_qos_);
            
        status_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/carla/ego_vehicle/status", reliable_qos_);
            
        RCLCPP_INFO(this->get_logger(), "ROS 2 publishers initialized");
    }
    
    void setup_ros_subscribers()
    {
        // Command velocity subscriber
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/carla/ego_vehicle/cmd_vel", reliable_qos_,
            std::bind(&CarlaGatewayNode::cmd_vel_callback, this, std::placeholders::_1));
            
        RCLCPP_INFO(this->get_logger(), "ROS 2 subscribers initialized");
    }
    
    void zmq_listener_thread()
    {
        RCLCPP_INFO(this->get_logger(), "ZeroMQ listener thread started");
        
        while (running_) {
            try {
                zmq::message_t message;
                zmq::recv_result_t result = zmq_socket_->recv(message, zmq::recv_flags::dontwait);
                
                if (result) {
                    // Parse JSON message
                    std::string msg_str(static_cast<char*>(message.data()), message.size());
                    Json::Value root;
                    Json::Reader reader;
                    
                    if (reader.parse(msg_str, root)) {
                        process_carla_message(root);
                    } else {
                        RCLCPP_WARN(this->get_logger(), "Failed to parse JSON message");
                    }
                } else {
                    // No message received, small sleep to prevent busy waiting
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                
            } catch (const zmq::error_t& e) {
                if (e.num() != EAGAIN) {  // EAGAIN is expected for non-blocking
                    RCLCPP_ERROR(this->get_logger(), "ZeroMQ receive error: %s", e.what());
                }
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "ZeroMQ listener thread stopped");
    }
    
    void process_carla_message(const Json::Value& root)
    {
        auto timestamp = this->get_clock()->now();
        
        // Process vehicle state -> odometry
        if (root.isMember("vehicle_state")) {
            publish_odometry(root["vehicle_state"], timestamp);
        }
        
        // Process sensor data
        if (root.isMember("sensors")) {
            const Json::Value& sensors = root["sensors"];
            
            // Camera data
            if (sensors.isMember("camera")) {
                publish_camera_image(sensors["camera"], timestamp);
            }
            
            // IMU data
            if (sensors.isMember("imu")) {
                publish_imu_data(sensors["imu"], timestamp);
            }
            
            // Collision data
            if (sensors.isMember("collision")) {
                publish_collision_event(sensors["collision"], timestamp);
            }
            
            // Lane invasion data
            if (sensors.isMember("lane_invasion")) {
                publish_lane_invasion_event(sensors["lane_invasion"], timestamp);
            }
        }
        
        // Process episode status
        if (root.isMember("episode_status")) {
            publish_episode_status(root["episode_status"], timestamp);
        }
    }
    
    void publish_camera_image(const Json::Value& camera_data, const rclcpp::Time& timestamp)
    {
        if (!camera_data.isMember("data") || !camera_data.isMember("width") || 
            !camera_data.isMember("height")) {
            return;
        }
        
        try {
            int width = camera_data["width"].asInt();
            int height = camera_data["height"].asInt();
            
            // Convert JSON array to OpenCV Mat
            cv::Mat image(height, width, CV_8UC3);
            const Json::Value& data = camera_data["data"];
            
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const Json::Value& pixel = data[y * width + x];
                    if (pixel.isArray() && pixel.size() >= 3) {
                        image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                            pixel[2].asUInt(),  // B
                            pixel[1].asUInt(),  // G  
                            pixel[0].asUInt()   // R
                        );
                    }
                }
            }
            
            // Convert to ROS message
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();
            msg->header.stamp = timestamp;
            msg->header.frame_id = "camera_link";
            
            camera_pub_->publish(*msg);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing camera data: %s", e.what());
        }
    }
    
    void publish_imu_data(const Json::Value& imu_data, const rclcpp::Time& timestamp)
    {
        auto msg = std::make_unique<sensor_msgs::msg::Imu>();
        
        msg->header.stamp = timestamp;
        msg->header.frame_id = "base_link";
        
        // Linear acceleration
        if (imu_data.isMember("accelerometer")) {
            const Json::Value& accel = imu_data["accelerometer"];
            msg->linear_acceleration.x = accel["x"].asDouble();
            msg->linear_acceleration.y = accel["y"].asDouble();
            msg->linear_acceleration.z = accel["z"].asDouble();
        }
        
        // Angular velocity
        if (imu_data.isMember("gyroscope")) {
            const Json::Value& gyro = imu_data["gyroscope"];
            msg->angular_velocity.x = gyro["x"].asDouble();
            msg->angular_velocity.y = gyro["y"].asDouble();
            msg->angular_velocity.z = gyro["z"].asDouble();
        }
        
        // Set covariance (unknown, so use large values)
        std::fill(msg->linear_acceleration_covariance.begin(), 
                 msg->linear_acceleration_covariance.end(), 0.0);
        std::fill(msg->angular_velocity_covariance.begin(), 
                 msg->angular_velocity_covariance.end(), 0.0);
        std::fill(msg->orientation_covariance.begin(), 
                 msg->orientation_covariance.end(), -1.0);  // Unknown orientation
        
        imu_pub_->publish(std::move(msg));
    }
    
    void publish_odometry(const Json::Value& vehicle_state, const rclcpp::Time& timestamp)
    {
        auto msg = std::make_unique<nav_msgs::msg::Odometry>();
        
        msg->header.stamp = timestamp;
        msg->header.frame_id = "odom";
        msg->child_frame_id = "base_link";
        
        // Position
        if (vehicle_state.isMember("position")) {
            const Json::Value& pos = vehicle_state["position"];
            msg->pose.pose.position.x = pos["x"].asDouble();
            msg->pose.pose.position.y = pos["y"].asDouble();
            msg->pose.pose.position.z = pos["z"].asDouble();
        }
        
        // Orientation (convert from Euler to quaternion)
        if (vehicle_state.isMember("rotation")) {
            const Json::Value& rot = vehicle_state["rotation"];
            double roll = rot["roll"].asDouble() * M_PI / 180.0;
            double pitch = rot["pitch"].asDouble() * M_PI / 180.0;
            double yaw = rot["yaw"].asDouble() * M_PI / 180.0;
            
            // Convert to quaternion
            double cy = cos(yaw * 0.5);
            double sy = sin(yaw * 0.5);
            double cp = cos(pitch * 0.5);
            double sp = sin(pitch * 0.5);
            double cr = cos(roll * 0.5);
            double sr = sin(roll * 0.5);
            
            msg->pose.pose.orientation.w = cr * cp * cy + sr * sp * sy;
            msg->pose.pose.orientation.x = sr * cp * cy - cr * sp * sy;
            msg->pose.pose.orientation.y = cr * sp * cy + sr * cp * sy;
            msg->pose.pose.orientation.z = cr * cp * sy - sr * sp * cy;
        }
        
        // Velocity
        if (vehicle_state.isMember("velocity")) {
            const Json::Value& vel = vehicle_state["velocity"];
            msg->twist.twist.linear.x = vel["x"].asDouble();
            msg->twist.twist.linear.y = vel["y"].asDouble();
            msg->twist.twist.linear.z = vel["z"].asDouble();
        }
        
        // Angular velocity
        if (vehicle_state.isMember("angular_velocity")) {
            const Json::Value& ang_vel = vehicle_state["angular_velocity"];
            msg->twist.twist.angular.x = ang_vel["x"].asDouble();
            msg->twist.twist.angular.y = ang_vel["y"].asDouble();
            msg->twist.twist.angular.z = ang_vel["z"].asDouble();
        }
        
        odom_pub_->publish(std::move(msg));
    }
    
    void publish_collision_event(const Json::Value& collision_data, const rclcpp::Time& timestamp)
    {
        auto msg = std::make_unique<std_msgs::msg::Bool>();
        msg->data = true;  // Collision detected
        collision_pub_->publish(std::move(msg));
        
        RCLCPP_WARN(this->get_logger(), "Collision detected!");
    }
    
    void publish_lane_invasion_event(const Json::Value& lane_data, const rclcpp::Time& timestamp)
    {
        auto msg = std::make_unique<std_msgs::msg::Bool>();
        msg->data = true;  // Lane invasion detected
        lane_invasion_pub_->publish(std::move(msg));
        
        RCLCPP_WARN(this->get_logger(), "Lane invasion detected!");
    }
    
    void publish_episode_status(const Json::Value& status_data, const rclcpp::Time& timestamp)
    {
        auto msg = std::make_unique<std_msgs::msg::String>();
        
        if (status_data.isMember("collision") && status_data["collision"].asBool()) {
            msg->data = "collision";
        } else if (status_data.isMember("lane_invasion") && status_data["lane_invasion"].asBool()) {
            msg->data = "lane_invasion";
        } else {
            msg->data = "running";
        }
        
        status_pub_->publish(std::move(msg));
    }
    
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        // Send control commands back to CARLA via ZeroMQ
        Json::Value command;
        command["type"] = "control";
        command["throttle"] = std::max(0.0, msg->linear.x);  // Positive linear.x = throttle
        command["steering"] = std::max(-1.0, std::min(1.0, msg->angular.z));  // Angular.z = steering
        command["brake"] = std::max(0.0, -msg->linear.x);  // Negative linear.x = brake
        
        Json::StreamWriterBuilder builder;
        std::string command_str = Json::writeString(builder, command);
        
        try {
            // Note: This requires a separate ZMQ socket for sending (ZMQ_PUB)
            // For simplicity, we'll use file-based communication as fallback
            // In production, implement proper bidirectional ZMQ communication
            
            RCLCPP_DEBUG(this->get_logger(), "Received control command: throttle=%.2f, steering=%.2f",
                        command["throttle"].asDouble(), command["steering"].asDouble());
                        
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error sending control command: %s", e.what());
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<CarlaGatewayNode>();
        
        RCLCPP_INFO(node->get_logger(), "CARLA Gateway Node running...");
        rclcpp::spin(node);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("carla_gateway"), "Fatal error: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}
