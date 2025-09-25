"""
@file uwb_server.py
@author Your Name
@brief Server for UWB Peer-to-Peer Localization System with EKF.
@version 0.1
@date 2023-10-27

@copyright Copyright (c) 2023

This script acts as the central server for a UWB localization system. It performs
the following key functions:
1.  **TDMA Coordination**: Implements a Time-Division Multiple Access (TDMA) scheme,
    dynamically assigning 'TAG' and 'ANCHOR' roles to connected ESP32 nodes to
    ensure orderly communication.
2.  **Data Reception**: Listens for UDP packets from the ESP32 nodes, containing
    UWB range data, IMU readings, and barometer data.
3.  **State Estimation**: Uses a sophisticated Extended Kalman Filter (EKF) for each
    node to fuse sensor data and estimate the 3D position and velocity of all nodes
    in the network. It performs a "joint" update, where a single range measurement
    improves the position estimate of *both* participating nodes.
4.  **Live Visualization**: Provides a real-time 3D plot of the estimated positions,
    orientations, and the distances between nodes using matplotlib.
"""

import socket
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
# Network settings for the server
LOCAL_IP = '10.21.65.119'  # The IP address of the computer running this script.
UDP_PORT_RX = 1234         # Port this server listens on for data from ESP32s.
UDP_PORT_TX = 1235         # Port this server sends commands to (ESP32s listen on this).

# IMPORTANT: Map the unique MAC addresses of your ESP32 devices to the static IP
# addresses you have assigned to them in your router's DHCP settings.
ESP32_IP_ADDRESSES = {
    "01:00:00:00:00:00:00:01": "192.168.137.70", # Example IP for Node 1
    "01:00:00:00:00:00:00:02": "192.168.137.228",# Example IP for Node 2
    # Add more nodes here following the format "MAC": "IP"
}

# TDMA (Time-Division Multiple Access) and EKF timing settings
RANGING_DURATION_S = 2.0   # Seconds each node will act as a 'TAG' in a cycle.
EKF_PREDICT_INTERVAL_S = 0.05 # How often to run the EKF prediction step if no new data arrives (50ms).

# Define the known initial starting positions for each node (in meters).
# This is crucial for the EKF to converge correctly.
INITIAL_POSITIONS = {
    "01:00:00:00:00:00:00:01": np.array([0.0, 0.0, 1.0]),
    "01:00:00:00:00:00:00:02": np.array([2.0, 0.0, 1.0]),
}

# ==============================================================================
# --- EKF AND NODE CLASSES ---
# ==============================================================================

class Node:
    """Represents a single UWB device in the network, managing its state via an EKF."""
    def __init__(self, mac_address, initial_pos=np.array([0.0, 0.0, 0.0])):
        """
        Initializes a node.
        @param mac_address: The unique MAC address of the node.
        @param initial_pos: The starting 3D position of the node.
        """
        self.mac_address = mac_address
        self.short_address = mac_address.split(':')[-1] # Used for cleaner display labels.

        # EKF State Vector: [px, py, pz, vx, vy, vz]' (position and velocity)
        self.x_hat = np.concatenate((initial_pos, np.array([0.0, 0.0, 0.0]))).reshape(-1, 1)
        # Covariance Matrix: Represents the uncertainty in the state estimate.
        self.P = np.eye(6) * 0.1

        # Timestamps for calculating delta_t (dt) in updates
        self.last_imu_time = time.time()
        self.last_ekf_update_time = time.time()
        self.last_range_time = {} # Tracks when the last range measurement with another node occurred

        # Simple yaw estimation from gyroscope for visualization
        self.yaw_estimate = 0.0 # Yaw in radians

    def predict(self, dt):
        """
        EKF Prediction Step. Estimates the future state based on a motion model.
        @param dt: Time delta since the last prediction.
        """
        # State transition matrix F (constant velocity model)
        F = np.array([
            [1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
        ])
        # Process noise Q: accounts for uncertainty in the motion model
        q_pos = 0.05 * dt**2; q_vel = 0.05 * dt
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        # Predict the next state and update the covariance
        self.x_hat = F @ self.x_hat
        self.P = F @ self.P @ F.T + Q
        self.last_ekf_update_time = time.time()

    def update_uwb_range(self, remote_node, measured_range, R_range=0.1):
        """
        EKF Update Step (Joint Update) using a UWB range measurement.
        This is a "joint" update because one measurement provides information
        about the positions of *both* nodes involved.
        @param remote_node: The other Node object involved in the ranging.
        @param measured_range: The distance measurement from the UWB hardware.
        @param R_range: The measurement noise variance for UWB.
        """
        # Get current positions from state vectors
        px1, py1, pz1 = self.x_hat[0,0], self.x_hat[1,0], self.x_hat[2,0]
        px2, py2, pz2 = remote_node.x_hat[0,0], remote_node.x_hat[1,0], remote_node.x_hat[2,0]

        # Calculate predicted range based on current state estimates
        predicted_range = np.sqrt((px1 - px2)**2 + (py1 - py2)**2 + (pz1 - pz2)**2)
        if predicted_range == 0: predicted_range = 0.001 # Avoid division by zero

        # Innovation (the difference between measured and predicted)
        y = measured_range - predicted_range

        # Jacobian of the measurement model (H) for both nodes
        Hx1 = np.array([(px1 - px2) / predicted_range, (py1 - py2) / predicted_range, (pz1 - pz2) / predicted_range, 0, 0, 0]).reshape(1, 6)
        Hx2 = np.array([(px2 - px1) / predicted_range, (py2 - py1) / predicted_range, (pz2 - pz1) / predicted_range, 0, 0, 0]).reshape(1, 6)

        # Create joint state, covariance, and Jacobian for the update
        x_joint = np.vstack((self.x_hat, remote_node.x_hat))
        P_joint = np.block([[self.P, np.zeros((6,6))], [np.zeros((6,6)), remote_node.P]])
        H_joint = np.hstack((Hx1, Hx2))
        R = np.array([[R_range]])

        # Kalman Filter standard equations
        S = H_joint @ P_joint @ H_joint.T + R
        K = P_joint @ H_joint.T @ np.linalg.inv(S)
        x_joint = x_joint + K @ np.array([[y]])
        P_joint = (np.eye(12) - K @ H_joint) @ P_joint

        # Decompose the joint state and covariance back into individual nodes
        self.x_hat, remote_node.x_hat = x_joint[0:6], x_joint[6:12]
        self.P, remote_node.P = P_joint[0:6, 0:6], P_joint[6:12, 6:12]

        # Constraint: Ensure Z position (height) is not negative
        self.x_hat[2,0] = max(0.0, self.x_hat[2,0])
        remote_node.x_hat[2,0] = max(0.0, remote_node.x_hat[2,0])

        # Update timestamp for this ranging pair
        self.last_range_time[remote_node.mac_address] = time.time()
        remote_node.last_range_time[self.mac_address] = time.time()

    def update_barometer(self, measured_pressure, R_pressure=0.01):
        """
        EKF Update Step using a barometer pressure measurement to estimate altitude (Z).
        @param measured_pressure: Pressure in hPa from the sensor.
        @param R_pressure: The measurement noise variance for the barometer.
        """
        # Simple barometric formula to convert pressure to altitude
        REFERENCE_PRESSURE_HPA = 938.0 # Calibrate this to your local conditions
        REFERENCE_ALTITUDE_M = 1.0    # Altitude at the reference pressure
        METERS_PER_HPA = 8.5          # Approximate conversion factor

        measured_altitude = REFERENCE_ALTITUDE_M + (REFERENCE_PRESSURE_HPA - measured_pressure) * METERS_PER_HPA

        # Innovation
        y = measured_altitude - self.x_hat[2,0]

        # Jacobian of the measurement model H (measures the 3rd state, pz)
        H = np.array([[0, 0, 1, 0, 0, 0]])
        R = np.array([[R_pressure]])

        # Kalman Filter standard equations
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ np.array([[y]])
        self.P = (np.eye(6) - K @ H) @ self.P

        # Constraint: Ensure Z position (height) is not negative
        self.x_hat[2,0] = max(0.0, self.x_hat[2,0])

    def update_imu(self, ax, ay, az, gx, gy, gz, dt):
        """
        Updates node orientation based on gyroscope data.
        Note: This is a simplified integration for visualization and not a full IMU fusion.
        @param gx, gy, gz: Gyroscope readings.
        @param dt: Time delta since the last IMU update.
        """
        # Integrate yaw rate (gy) to get yaw angle
        self.yaw_estimate += gy * dt
        self.last_imu_time = time.time()

# ==============================================================================
# --- VISUALIZATION ---
# ==============================================================================
class LivePlotter:
    """Handles the real-time 3D visualization of the UWB nodes."""
    def __init__(self):
        """Initializes the matplotlib figure and 3D axes."""
        plt.ion() # Turn on interactive mode
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)'); self.ax.set_zlabel('Z (m)')
        self.ax.set_title('UWB Peer-to-Peer EKF Localization')
        self.ax.set_aspect('equal', adjustable='box')
        self.scatter = self.ax.scatter([], [], [], s=150)

    def update(self, nodes, active_tagger_id):
        """
        Redraws the plot with the latest node data.
        @param nodes: A dictionary of all Node objects.
        @param active_tagger_id: The MAC address of the current node acting as the TAG.
        """
        if not nodes: return

        positions = np.array([node.x_hat[0:3].flatten() for node in nodes.values()])
        if positions.ndim == 1: positions = positions.reshape(1, -1)
        node_ids = list(nodes.keys())

        # Clear and re-configure axes for each frame
        self.ax.clear()
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)'); self.ax.set_zlabel('Z (m)')
        self.ax.set_title(f'UWB EKF Localization (Tagger: {active_tagger_id.split(":")[-1] if active_tagger_id else "N/A"})')
        self.ax.set_aspect('equal', adjustable='box')

        # Auto-scale plot limits
        if not positions.any():
            self.ax.set_xlim(-5, 5); self.ax.set_ylim(-5, 5); self.ax.set_zlim(0, 5)
        else:
            min_c, max_c = positions.min(axis=0) - 1.5, positions.max(axis=0) + 1.5
            self.ax.set_xlim(min_c[0], max_c[0]); self.ax.set_ylim(min_c[1], max_c[1]); self.ax.set_zlim(min_c[2], max_c[2] + 1.5)

        # Draw nodes (red for tagger, blue for anchors)
        colors = ['red' if node_id == active_tagger_id else 'blue' for node_id in node_ids]
        self.scatter = self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=150, c=colors, edgecolors='black', depthshade=True)

        # Draw labels, orientation arrows, and ranging links
        for i, node_id in enumerate(node_ids):
            node = nodes[node_id]
            pos = positions[i]
            short_id = node_id.split(':')[-1]

            # Text label for each node
            label = f" {short_id}\n Z:{pos[2]:.2f}m\n Yaw:{np.degrees(node.yaw_estimate):.0f}°"
            self.ax.text(pos[0], pos[1], pos[2] + 0.2, label, color='black', fontsize=8, ha='center', va='bottom')

            # Orientation arrow (based on yaw)
            arrow_length = 0.5
            dx = arrow_length * np.cos(node.yaw_estimate)
            dy = arrow_length * np.sin(node.yaw_estimate)
            self.ax.quiver(pos[0], pos[1], pos[2], dx, dy, 0, color='green', linewidth=2)

            # Draw lines for recent range measurements
            for remote_id, last_range_time in node.last_range_time.items():
                if time.time() - last_range_time < (RANGING_DURATION_S * 2): # Show recent links
                    if remote_id in nodes:
                        remote_node_index = node_ids.index(remote_id)
                        remote_pos = positions[remote_node_index]
                        if node_id < remote_id: # Draw each link only once
                            self.ax.plot([pos[0], remote_pos[0]], [pos[1], remote_pos[1]], [pos[2], remote_pos[2]], 'k--', alpha=0.6, linewidth=1)
                            # Display distance on the link
                            mid_point = (pos + remote_pos) / 2
                            actual_distance = np.linalg.norm(pos - remote_pos)
                            self.ax.text(mid_point[0], mid_point[1], mid_point[2] + 0.1, f'{actual_distance:.2f}m', color='purple', fontsize=7, ha='center')

        plt.draw()
        plt.pause(0.001)

# =======================================================================================
# --- UDP SERVER AND MAIN LOGIC ---
# =======================================================================================
def send_role_command(ip_address, role):
    """Sends a role command ('ROLE:TAG' or 'ROLE:ANCHOR') to an ESP32."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock_tx:
            sock_tx.sendto(f"ROLE:{role}".encode(), (ip_address, UDP_PORT_TX))
    except Exception as e:
        print(f"Error sending command to {ip_address}: {e}")

if __name__ == "__main__":
    # Initialize all configured nodes
    nodes = {mac: Node(mac, INITIAL_POSITIONS.get(mac, np.array([0.0, 0.0, 0.0]))) for mac in ESP32_IP_ADDRESSES.keys()}
    print("Initialized nodes:", ", ".join(nodes.keys()))

    # Setup UDP socket for receiving data
    sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx.bind((LOCAL_IP, UDP_PORT_RX))
    sock_rx.settimeout(EKF_PREDICT_INTERVAL_S) # Timeout for non-blocking receive

    # Initialize plotter and TDMA variables
    plotter = LivePlotter()
    current_tagger_index = 0
    node_mac_list = list(ESP32_IP_ADDRESSES.keys())
    node_roles = {mac: "IDLE" for mac in node_mac_list}

    print("Server started. Beginning TDMA scheduling...")

    try:
        # Main server loop
        while True:
            if not node_mac_list:
                print("No nodes configured. Please check ESP32_IP_ADDRESSES."); time.sleep(1); continue

            # --- TDMA SCHEDULING ---
            # Select the next node to be the tagger
            tagger_mac = node_mac_list[current_tagger_index]
            print(f"\n--- Cycle Start: Assigning {tagger_mac.split(':')[-1]} as Tagger ---")

            # Send role commands to all nodes
            for mac, ip in ESP32_IP_ADDRESSES.items():
                new_role = "TAG" if mac == tagger_mac else "ANCHOR"
                if node_roles.get(mac) != new_role:
                    send_role_command(ip, new_role)
                    node_roles[mac] = new_role

            time.sleep(0.5) # Give nodes time to switch roles

            # --- DATA COLLECTION AND PROCESSING ---
            listen_end_time = time.time() + RANGING_DURATION_S
            while time.time() < listen_end_time:
                try:
                    # Listen for incoming data packets
                    data, addr = sock_rx.recvfrom(512)
                    decoded_data = data.decode().strip()
                    parts = decoded_data.split(',')

                    # Ensure packet is well-formed
                    if len(parts) == 10:
                        mac, remote_short, ax, ay, az, gx, gy, gz, pressure, uwb_range = parts
                        if mac in nodes:
                            node = nodes[mac]
                            # Convert string data to floats
                            ax, ay, az, gx, gy, gz, pressure, uwb_range = map(float, [ax, ay, az, gx, gy, gz, pressure, uwb_range])

                            # Find the full MAC address of the remote node from its short address
                            remote_mac_full = next((k for k in ESP32_IP_ADDRESSES if k.endswith(remote_short.upper())), None)

                            # IMU and Barometer Update
                            dt_imu = time.time() - node.last_imu_time
                            if dt_imu > 0:
                                node.update_imu(ax, ay, az, gx, gy, gz, dt_imu)
                                node.update_barometer(pressure)

                            # UWB Range Update (if valid)
                            if uwb_range > 0 and remote_mac_full and remote_mac_full in nodes:
                                remote_node = nodes[remote_mac_full]
                                node.update_uwb_range(remote_node, uwb_range)
                                print(f"Range Update: {node.short_address} <-> {remote_node.short_address} = {uwb_range:.2f}m")

                except socket.timeout:
                    # No data received, just continue the loop
                    pass
                except Exception as e:
                    print(f"Error during data reception/processing: {e}")

            # --- EKF PREDICTION & VISUALIZATION ---
            # Run prediction for all nodes to account for motion during the last interval
            for node in nodes.values():
                dt_predict = time.time() - node.last_ekf_update_time
                node.predict(dt_predict)

            # Update the 3D plot
            plotter.update(nodes, tagger_mac)

            # Move to the next node in the TDMA cycle
            current_tagger_index = (current_tagger_index + 1) % len(node_mac_list)
            time.sleep(0.1) # Brief pause between TDMA cycles

    except KeyboardInterrupt:
        print("\nServer shutdown initiated by user.")
    except Exception as e:
        print(f"\nAn unhandled error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock_rx.close()
        plt.ioff() # Turn off interactive mode
        plt.show() # Keep the final plot window open
        print("Server shutdown complete.")