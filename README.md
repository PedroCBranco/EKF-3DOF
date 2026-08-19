# Extended Kalman Filter (EKF) - 3DOF Sensor Fusion 

### Overview
This project implements an Extended Kalman Filter (EKF) from scratch in Python. It was developed to accurately estimate the state of a 3 Degrees of Freedom (3DOF) vehicle (X, Y, theta) by fusing high-frequency IMU data (control inputs) with lower-frequency, noisy GPS and Compass readings.

### Key Features
* **Custom Mathematical Model:** Built without high-level estimation libraries to deeply understand the underlying matrix algebra and probabilistic robotics.
* **Multi-rate Sensor Fusion:** Simulates real-world hardware constraints where the IMU (prediction step) operates at a higher frequency than the GPS (correction step).
* **Outlier Rejection (Mahalanobis):** Implements a statistical threshold using the Mahalanobis distance to reject anomalous GPS jumps, ensuring filter stability.
* **Non-Linear Dynamics:** Utilizes Jacobian matrices (Taylor Series expansion) to linearize the system's trigonometry (sin and cos) at every operating point.

### Why Python & 3DOF?
As a foundational project, Python was chosen to allow rapid mathematical prototyping and matrix debugging before moving to C++ for embedded systems. The 3DOF constraint isolates the planar navigation logic, keeping the Jacobian manageable and focusing entirely on the filter's architecture.

### How to Run
Ensure you have `numpy` and `matplotlib` installed.
```bash
py main.py
