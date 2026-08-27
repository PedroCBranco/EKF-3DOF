import numpy as np
from ekf import EKF
from graphs import gerar_relatorio

#----------------------------------SIMULATION----------------------------------

dt = 0.1
tempo_passado = []

#  Initial state of the system in reality

posicao_real_x = 0.0      
posicao_real_y = 0.0
velocidade_real = 0.0
teta_real = 0.0
 

incerteza_gps = 2.0     
incerteza_bussola = 0.1
incerteza_imu_a = 0.05
incerteza_imu_w = 0.005

historico_realx, historico_sensorx, historico_previsaox = [], [], []
historico_realy, historico_sensory, historico_previsaoy = [], [], []


# Initialization of the filter

ekf1 = EKF(dt= dt )

#Simulate different behaviour at different times

for i in range(300):    

    if (i <= 100):
        aceleracao_real = 1.0
        w_real = 0.0
    elif (100 < i <= 200):
        aceleracao_real = 0.0
        w_real = 0.3
    elif (200 < i <= 300):
        aceleracao_real = -1.0
        w_real = 0.0

    momento_atual = i * dt
    tempo_passado.append(momento_atual)

    # Theorical model of the physics of the system

    posicao_real_x += velocidade_real * np.cos(teta_real) * dt
    posicao_real_y += velocidade_real * np.sin(teta_real) * dt
    velocidade_real += aceleracao_real * dt
    teta_real += w_real * dt                                               
    
    # Readings from the sensors with added gaussian/white noise

    aceleracao_medida = aceleracao_real + np.random.normal(0, incerteza_imu_a)
    w_medido = w_real + np.random.normal(0, incerteza_imu_w)
    
    # Input command given to the car

    u = np.array([[aceleracao_medida], 
                  [w_medido]])

    #------------------------PREDICTION STEP--------------------------
    ekf1.prediction_step(u)



    # ----------------------CORRECTION STEP---------------------------

    gps_x = np.nan         
    gps_y = np.nan


    # We make the frequency of the compass and gps slower then that of the imu, like it is in reality (sensor fusion), also adding gaussing/white noise

    if (i % 10 == 0): 
        
        gps_x = posicao_real_x + np.random.normal(0, incerteza_gps)               
        gps_y = posicao_real_y + np.random.normal(0, incerteza_gps)
        bussola_teta = teta_real + np.random.normal(0, incerteza_bussola)

        z = np.array([[gps_x],                   # Measurement vector
                      [gps_y],
                      [bussola_teta]])
        
        ekf1.correction_step(z)
        

    historico_realx.append(posicao_real_x)
    historico_sensorx.append(gps_x)
    historico_previsaox.append(ekf1.x[0, 0])

    historico_realy.append(posicao_real_y)
    historico_sensory.append(gps_y)
    historico_previsaoy.append(ekf1.x[1, 0])

gerar_relatorio(tempo_passado, historico_realx, historico_realy, 
                historico_sensorx, historico_sensory, 
                historico_previsaox, historico_previsaoy)
