import numpy as np
import matplotlib.pyplot as plt

def gerar_relatorio(tempo_passado, historico_realx, historico_realy, 
                    historico_sensorx, historico_sensory, 
                    historico_previsaox, historico_previsaoy):
    
    #------------------------GRAPHS--------------------------------------
    real_x = np.array(historico_realx)
    previsao_x = np.array(historico_previsaox)
    real_y = np.array(historico_realy)
    previsao_y = np.array(historico_previsaoy)

    erro_instantaneo = np.sqrt((real_x - previsao_x)**2 + (real_y - previsao_y)**2)

    rmse_total = np.sqrt(np.mean((real_x - previsao_x)**2 + (real_y - previsao_y)**2))
    print(f"\n--- RELATÓRIO DO EKF ---")
    print(f"RMSE Total da Trajetória: {rmse_total:.2f} metros\n")


    plt.title("Gráfico posição")
    plt.xlabel("Posição X")
    plt.ylabel("Posição Y")

    plt.plot(historico_realx, historico_realy, label="Trajetória Real", color="#fc0839", )
    plt.plot(historico_sensorx, historico_sensory, marker=".", linestyle="None", color="#08fc4d", markersize=8, label="GPS")
    plt.plot(historico_previsaox, historico_previsaoy, label="Filtro de Kalman (EKF)", color="#08cbfc")

    plt.legend()
    plt.grid(True)
    plt.axis('equal') 
    plt.show()


    plt.title("Evolução do Erro de Posição")
    plt.xlabel("Tempo (segundos)")
    plt.ylabel("Erro (metros)")
    plt.plot(tempo_passado, erro_instantaneo, label="Erro Absoluto (EKF)", color="#fc0839")
    plt.grid(True)
    plt.legend()
    plt.show()