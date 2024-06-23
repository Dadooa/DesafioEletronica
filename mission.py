#!/usr/bin/env python

import rospy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandTOL, SetMode, CommandBool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from detect import CrossDetection  # Importa a classe de detecção de bases

current_state = State()
bridge = CvBridge()
cross_detector = CrossDetection()

# Variáveis globais para controle da missão
BASES_FIXAS = [(0, 5.5, 4), (12, 0, 4)]  # Posições das bases fixas (x, y, z)
BASES_FIXAS_VISITADAS = []  # Lista para armazenar as bases fixas visitadas
BASES_MOVEIS = []  # Lista para armazenar as posições das bases móveis
ORIGEM = (0, 0, 4)  # Posição de origem do drone (x, y, z)
ARENA_LADO = 12  # Lado da arena quadrada
NUM_QUADRADOS = 5  # Número de quadrados a contornar
ALTITUDE_DECOLAGEM = 4  # Altitude inicial de decolagem

# Parâmetros da câmera (definidos diretamente)
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)  # Sem distorção

def state_cb(state):
    global current_state
    current_state = state

def is_near_fixed_base(base_position, fixed_bases, tolerance=1.0):
    for fixed_base in fixed_bases:
        distance = np.linalg.norm(np.array(base_position) - np.array(fixed_base))
        if distance <= tolerance:
            return True
    return False

def image_to_world(uv, camera_matrix, dist_coeffs, altitude):
    uv = np.array(uv, dtype=np.float32)
    uv = np.reshape(uv, (-1, 1, 2))
    undistorted_points = cv2.undistortPoints(uv, camera_matrix, dist_coeffs, None, camera_matrix)
    undistorted_points = undistorted_points[0, 0, :]
    x_world = (undistorted_points[0] - camera_matrix[0, 2]) * altitude / camera_matrix[0, 0]
    y_world = (undistorted_points[1] - camera_matrix[1, 2]) * altitude / camera_matrix[1, 1]
    z_world = 0
    return (x_world, y_world, z_world)

def image_callback(data):
    global BASES_MOVEIS, BASES_FIXAS_VISITADAS
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        
        # Chama a função de detecção de base
        list_of_bases_fixed, list_of_bases_moving = cross_detector.base_detection(cv_image, BASES_FIXAS)

        # Tolerância para considerar uma base próxima de uma base fixa
        tolerance = 1.0  # Ajuste conforme necessário

        # Armazena as posições das bases móveis detectadas
        for base in list_of_bases_moving:
            world_position = image_to_world((base[0], base[1]), camera_matrix, dist_coeffs, ALTITUDE_DECOLAGEM)
            if not is_near_fixed_base(world_position, BASES_FIXAS_VISITADAS, tolerance):
                BASES_MOVEIS.append(world_position)
                rospy.loginfo(f"Base móvel detectada na posição: {world_position}")

        # Armazena as posições das bases fixas visitadas
        for base in list_of_bases_fixed:
            world_position = image_to_world((base[0], base[1]), camera_matrix, dist_coeffs, ALTITUDE_DECOLAGEM)
            if world_position not in BASES_FIXAS_VISITADAS:
                BASES_FIXAS_VISITADAS.append(world_position)
                rospy.loginfo(f"Base fixa visitada na posição: {world_position}")

    except CvBridgeError as e:
        rospy.logerr("Erro ao converter imagem: %s" % e)

def set_mode(mode):
    rospy.wait_for_service('/edrn/mavros/set_mode')
    try:
        set_mode_service = rospy.ServiceProxy('/edrn/mavros/set_mode', SetMode)
        response = set_mode_service(base_mode=0, custom_mode=mode)
        if response.mode_sent:
            rospy.loginfo(f"Modo alterado para {mode}")
        else:
            rospy.logwarn("Falha ao alterar o modo.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Falha ao chamar o serviço set_mode: {e}")

def armar_drone():
    rospy.wait_for_service('/edrn/mavros/cmd/arming')
    try:
        arming_service = rospy.ServiceProxy('/edrn/mavros/cmd/arming', CommandBool)
        response = arming_service(value=True)  # Arma o drone
        if response.success:
            rospy.loginfo("Drone armado com sucesso!")
        else:
            rospy.logwarn("Falha ao armar o drone.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Erro ao chamar o serviço de armar: {e}")
    rospy.sleep(10)

def decolar(altitude):
    rospy.wait_for_service('/edrn/mavros/cmd/takeoff')
    try:
        takeoff_service = rospy.ServiceProxy('/edrn/mavros/cmd/takeoff', CommandTOL)
        takeoff_service(altitude=altitude)  
        rospy.loginfo(f"Decolando para {altitude} metros de altitude")
    except rospy.ServiceException as e:
        rospy.logerr(f"Falha ao chamar o serviço de decolagem: {e}")

def mover_drone(x, y, z):
    rospy.loginfo(f"Movendo o drone para a posição ({x}, {y}, {z})...")
    publisher_posicao = rospy.Publisher('/edrn/mavros/setpoint_position/local', PoseStamped, queue_size=10)
    rate = rospy.Rate(20)  # Frequência de publicação

    posicao = PoseStamped()
    posicao.header.stamp = rospy.Time.now()
    posicao.pose.position.x = x
    posicao.pose.position.y = y
    posicao.pose.position.z = z

    # Publica a posição desejada até que a mudança seja confirmada
    for i in range(100):
        publisher_posicao.publish(posicao)
        rate.sleep()

    rospy.loginfo(f"Posição do drone atualizada para x={x}, y={y}, z={z}")

def iniciar_captura_camera():
    rospy.Subscriber("/edrn/camera/color/image_raw", Image, image_callback)

def pousar_drone():
    rospy.wait_for_service('/edrn/mavros/cmd/land')
    try:
        land_service = rospy.ServiceProxy('/mavros/cmd/land', CommandTOL)
        response = land_service()  # Comando para pousar o drone
        if response.success:
            rospy.loginfo("Drone pousando...")
        else:
            rospy.logwarn("Falha ao pousar o drone.")
    except rospy.ServiceException as e:
        rospy.logerr("Falha ao chamar o serviço de pouso: %s" % e)

def calcular_posicoes_contorno_arena(lado_arena, num_quadrados):
    posicoes = []

    lado_atual = lado_arena
    passo = lado_arena / num_quadrados
    x_min, x_max = -lado_atual / 2, lado_atual / 2
    y_min, y_max = -lado_atual / 2, lado_atual / 2

    for i in range(num_quadrados):
        posicoes.append((x_min, y_min, ALTITUDE_DECOLAGEM))
        posicoes.append((x_max, y_min, ALTITUDE_DECOLAGEM))
        posicoes.append((x_max, y_max, ALTITUDE_DECOLAGEM))
        posicoes.append((x_min, y_max, ALTITUDE_DECOLAGEM))

        lado_atual -= passo
        x_min += passo / 2
        x_max -= passo / 2
        y_min += passo / 2
        y_max -= passo / 2

    return posicoes

def ordenar_bases_por_proximidade(posicao_atual, bases):
    bases_ordenadas = sorted(bases, key=lambda base: distancia_euclidiana(posicao_atual, base))
    return bases_ordenadas

def distancia_euclidiana(posicao1, posicao2):
    return np.sqrt((posicao1[0] - posicao2[0])**2 + (posicao1[1] - posicao2[1])**2)

def main():
    rospy.init_node('controle_drone', anonymous=True)

    rospy.Subscriber("/edrn/mavros/state", State, state_cb)
    iniciar_captura_camera()

    set_mode("GUIDED")
    armar_drone()
    decolar(ALTITUDE_DECOLAGEM)
    
    while not current_state.mode == "GUIDED" and not rospy.is_shutdown():
        rospy.loginfo("Aguardando que o drone entre no modo GUIDED...")
        rospy.sleep(1)

    # Navegar até as bases fixas
    for base in BASES_FIXAS:
        mover_drone(base[0], base[1], base[2])
        rospy.loginfo(f"Drone alcançou a base fixa em ({base[0]}, {base[1]}, {base[2]})")
        BASES_FIXAS_VISITADAS.append((base[0], base[1], base[2]))

    # Contornar a arena em quadrados
    posicoes_contorno = calcular_posicoes_contorno_arena(ARENA_LADO, NUM_QUADRADOS)
    for posicao in posicoes_contorno:
        mover_drone(posicao[0], posicao[1], posicao[2])
        rospy.loginfo(f"Drone contornou a arena na posição ({posicao[0]}, {posicao[1]}, {posicao[2]})")

    # Retorno à origem sem pousar
    mover_drone(ORIGEM[0], ORIGEM[1], ORIGEM[2])
    rospy.loginfo("Drone retornou à origem sem pousar")

    # Visita às bases móveis em ordem de proximidade
    bases_ordenadas = ordenar_bases_por_proximidade(ORIGEM, BASES_MOVEIS)
    for base in bases_ordenadas:
        mover_drone(base[0], base[1], base[2])
        pousar_drone()
        armar_drone()
        decolar(ALTITUDE_DECOLAGEM)
        rospy.loginfo(f"Drone alcançou a base móvel em ({base[0]}, {base[1]}, {base[2]})")

    rospy.spin()

if __name__ == '__main__':
    main()
