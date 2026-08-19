import numpy as np
from numpy.linalg import inv

# This class receives multiple parameters needed for the prediction and correction steps pf the EKF, in light of that it has three processes,
# an initial one for creating the matrixes of the respective physical model, and two other ones, respectivly the prediction and correction steps

class EKF:
    def __init__(self, dt, posX=0.0, posY=0.0, V=0.0, teta=0.0,
                 incerteza_pos=4.0, incerteza_vel=4.0, incerteza_teta=0.1):
        
        self.dt = dt
        
        # State vector, composed by the positions, linear velocity and angle of rotation 
        self.x = np.array([[posX], 
                           [posY],
                           [V],
                           [teta]])
        
        # Inital uncertainty matrix, contains the variances and covariances between variables
        self.P_x = np.array([[incerteza_pos, 0.0, 0.0, 0.0],
                             [0.0, incerteza_pos, 0.0, 0.0],
                             [0.0, 0.0, incerteza_vel, 0.0],
                             [0.0, 0.0, 0.0, incerteza_teta]])
        
        # Process noise matrix, represents our trust in the physical model (Consideration for slips, bumps, or friction)  
        self.Q_x = np.eye(4) * 0.01

        # Observation Matrix, only contais a map of which variables are eventually measured, marked with a one, otherwise that line is omited (like linear velocity here)
        self.H_x = np.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
        
        # Sensor Noise matrix, holds the variance os the variables which are measured
        self.R_x = np.array([[2.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0],
                             [0.0, 0.0, 0.1]]) 
        
    def prediction_step(self, u):
       # In the prediction step we received all the written beforehand matrixes and the command vector u, which hols both linear and angular accelaration
       
        a = u[0, 0]
        w = u[1, 0]
        
        V_antigo = self.x[2, 0]
        teta_antigo = self.x[3, 0]
        
        self.x[0, 0] = self.x[0, 0] + V_antigo * np.cos(teta_antigo) * self.dt
        self.x[1, 0] = self.x[1, 0] + V_antigo * np.sin(teta_antigo) * self.dt
        self.x[2, 0] = V_antigo + a * self.dt
        self.x[3, 0] = teta_antigo + w * self.dt
        
        # Jacobian, holds the partial derivatives of the equations whom describe the behaviour of the system
        # Its initialized here and not before because it need the current velocity and angle and not the inital ones
        
        F = np.array([[1.0, 0.0, np.cos(teta_antigo)*self.dt, -V_antigo*np.sin(teta_antigo)*self.dt],
                      [0.0, 1.0, np.sin(teta_antigo)*self.dt,  V_antigo*np.cos(teta_antigo)*self.dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
                      
        # Uncertainty propagation
        self.P_x = F @ self.P_x @ F.T + self.Q_x

    def correction_step(self, z):
        # In the correction step we receive what we had before and the measurents done by the sensors, held in measurement vector z
        erro_x = z - (self.H_x @ self.x)
        S_x = (self.H_x @ self.P_x @ self.H_x.T) + self.R_x

        # --- OUTLIER REJECTION (Mahalanobis Gating)
        distancia_mahalanobis = erro_x.T @ inv(S_x) @ erro_x
        limite_aceitavel = 7

        if distancia_mahalanobis > limite_aceitavel:
            return 
        
        Kalman_gain_x = (self.P_x @ self.H_x.T) @ inv(S_x)
        
        self.x = self.x + (Kalman_gain_x @ erro_x)
        self.P_x = (np.eye(4) - (Kalman_gain_x @ self.H_x)) @ self.P_x