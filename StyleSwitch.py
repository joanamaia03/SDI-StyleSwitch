import cv2
import numpy as np
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)

dress_images_women = [
    cv2.imread('Images/mulher1.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('Images/mulher2.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('Images/mulher3.png', cv2.IMREAD_UNCHANGED),
]

dress_images_men = [
    cv2.imread('Images/homem1.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('Images/homem2.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('Images/homem3.png', cv2.IMREAD_UNCHANGED),
]

dress_images = dress_images_women 
current_dress_index = 0

cap = cv2.VideoCapture(0)

# Botões para mudar o sexo (as cores RGB estão ao contrário -> GBR)
def draw_buttons(frame):
    button_women_pos = (frame.shape[1] - 100, 50)
    button_men_pos = (50, 50) 

    cv2.rectangle(frame, (button_women_pos[0] - 10, button_women_pos[1] - 10), (button_women_pos[0] + 30, button_women_pos[1] + 20), (219, 203, 255), -1)
    cv2.rectangle(frame, (button_men_pos[0] - 10, button_men_pos[1] - 10), (button_men_pos[0] + 30, button_men_pos[1] + 20), (222, 176, 56), -1)  

    cv2.putText(frame, 'F', (button_women_pos[0], button_women_pos[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, 'M', (button_men_pos[0], button_men_pos[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(frame, (button_women_pos[0] - 10, button_women_pos[1] - 10), (button_women_pos[0] + 30, button_women_pos[1] + 20), (104, 35, 198), 2) 
    cv2.rectangle(frame, (button_men_pos[0] - 10, button_men_pos[1] - 10), (button_men_pos[0] + 30, button_men_pos[1] + 20), (63, 13, 2), 2) 
    
    return button_men_pos, button_women_pos

prev_hand_position = None
current_dress_index = 0

last_change_time = 0
change_delay = 1 

cap_is_opened = True
selected_gender = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converte a imagem para RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    button_men_pos, button_women_pos = draw_buttons(frame)

    # Deteta se a mão direita altera de posição
    if results.pose_landmarks:
        right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        hand_x = int(right_hand.x * frame.shape[1])
        hand_y = int(right_hand.y * frame.shape[0])

        # Verifica se a mão está sobre os botões
        if (button_men_pos[0] - 10 <= hand_x <= button_men_pos[0] + 30) and (button_men_pos[1] - 10 <= hand_y <= button_men_pos[1] + 20):
            dress_images = dress_images_men
            selected_gender = True
        elif (button_women_pos[0] - 10 <= hand_x <= button_women_pos[0] + 30) and (button_women_pos[1] - 10 <= hand_y <= button_women_pos[1] + 20):
            dress_images = dress_images_women
            selected_gender = True

         # Verifica se a palma da mão está levantada
        palm_open = False
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].visibility > 0.5:
            index_finger = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
            index_x = int(index_finger.x * frame.shape[1])
            index_y = int(index_finger.y * frame.shape[0])

            if hand_y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frame.shape[0]:  
                palm_open = True

        # Muda de roupa 
        current_time = time.time()  
        if palm_open and (current_time - last_change_time) > change_delay:
            current_dress_index = (current_dress_index + 1) % len(dress_images)
            last_change_time = current_time

        if selected_gender:  
            prev_hand_position = (hand_x, hand_y)
            current_dress = dress_images[current_dress_index]

            dress_height = 500 
            shoulder_width = 450  

            resized_dress = cv2.resize(current_dress, (shoulder_width, dress_height))

            # Posiciona o vestido em relação ao nariz
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            nose_x = int(nose.x * frame.shape[1])
            nose_y = int(nose.y * frame.shape[0])

            dress_y_start = nose_y + 20  
            dress_x_start = nose_x - shoulder_width // 2 + 10
            dress_x_end = nose_x + shoulder_width // 2 + 10

            # Posiciona o vestido no feed da câmara
            if dress_x_start < 0:
                dress_x_start_crop = abs(dress_x_start)
                dress_x_start = 0
            else:
                dress_x_start_crop = 0

            if dress_x_end > frame.shape[1]:
                dress_x_end_crop = shoulder_width - (dress_x_end - frame.shape[1]) 
                dress_x_end = frame.shape[1]
            else:
                dress_x_end_crop = shoulder_width

            if dress_y_start + dress_height > frame.shape[0]:
                dress_y_end_crop = frame.shape[0] - dress_y_start 
            else:
                dress_y_end_crop = dress_height

            dress_region = frame[dress_y_start:dress_y_start + dress_height, dress_x_start:dress_x_end]

            if resized_dress.shape[0] == dress_height and resized_dress.shape[1] == (dress_x_end - dress_x_start):
                cropped_dress = resized_dress[0:dress_y_end_crop, dress_x_start_crop:dress_x_end_crop]
                dress_rgb = resized_dress[..., :3]
                dress_alpha = resized_dress[..., 3] / 255.0  

                min_height = min(dress_region.shape[0], dress_rgb.shape[0])
                min_width = min(dress_region.shape[1], dress_rgb.shape[1])

                dress_region = dress_region[:min_height, :min_width]
                dress_rgb = dress_rgb[:min_height, :min_width]
                dress_alpha = dress_alpha[:min_height, :min_width]

                for c in range(0, 3):
                    dress_region[..., c] = dress_alpha * dress_rgb[..., c] + (1.0 - dress_alpha) * dress_region[..., c]

                frame[dress_y_start:dress_y_start + dress_height, dress_x_start:dress_x_end] = dress_region

    cv2.imshow('Dress', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()