"""
Модуль с функиями для предикта сигарет по фото и видео
"""
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Размер к которому приводить изображение
IMG_SIZE = 1024
# Частота кадров
FPS = 25.0


# Функция загружает с диска модель с весами на указанное устройство
def get_model(model_PATH, device_name=''):
    if device_name == '':
        model = load_model(model_PATH)
    else:
        with tf.device(device_name):
            model = load_model(model_PATH)
    return model


# Функция предикта одной картинки
def img_predict(model, img_FILE, out_FILE):
    curr_image = cv2.imread(img_FILE)
    # переходим к RGB
    # curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
    # сохраним оригинальные размеры картинки
    curr_w = curr_image.shape[1]
    curr_h = curr_image.shape[0]
    # рассчиитаем коэффициен для изменения размера
    if curr_w > curr_h:
        scale_frame = IMG_SIZE / curr_w
    else:
        scale_frame = IMG_SIZE / curr_h
    # и новые размеры изображения
    new_width = int(curr_w * scale_frame)
    new_height = int(curr_h * scale_frame)
    # делаем ресайз к целевым размерам
    curr_image = cv2.resize(curr_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Подготовим картинку для подачи в нейронку
    data = cv2.resize(curr_image, (416, 416), interpolation=cv2.INTER_AREA)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = np.asarray(data)
    data = data / 255.
    image_to_pred = np.expand_dims(data, axis=0)
    prediction = model.predict(image_to_pred)
    p_xmin, p_ymin, p_xmax, p_ymax = prediction[0]

    # Пересчитаем в пикселы для фрейма
    p_xmin = int(p_xmin * new_width)
    p_ymin = int(p_ymin * new_height)
    p_xmax = int(p_xmax * new_width)
    p_ymax = int(p_ymax * new_height)

    # Нарисуем найденный bb на картинке
    cv2.rectangle(curr_image, (p_xmin, p_ymin), (p_xmax, p_ymax), (0, 0, 255), 2)
    # Сохраним картинку в целевую папку
    cv2.imwrite(out_FILE, curr_image)

    # имя и путь выходного файла сейчас не меняются
    return out_FILE

# Функция предикта видео
def vid_predict(model, vid_FILE, out_FILE):
    # открываем видео
    cap = cv2.VideoCapture(vid_FILE)
    if (cap.isOpened() == False):
        print("Ошибка открытия файла видео")

    # Рассчитаем коэффициент для изменения размера
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if width > height:
        scale_frame = IMG_SIZE / width
    else:
        scale_frame = IMG_SIZE / height
    # и новые размеры фрейма
    new_width = int(width * scale_frame)
    new_height = int(height * scale_frame)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(out_FILE, fourcc, FPS, (new_width, new_height))

    # Получаем фреймы пока видео не закончится
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            # Изменим размер фрейма
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # Подготовим картинку для подачи в нейронку
            data = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_AREA)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data = np.asarray(data)
            data = data / 255.
            frame_to_pred = np.expand_dims(data, axis=0)
            prediction = model.predict(frame_to_pred)
            p_xmin, p_ymin, p_xmax, p_ymax = prediction[0]

            # Пересчитаем в пикселы для фрейма
            p_xmin = int(p_xmin * new_width)
            p_ymin = int(p_ymin * new_height)
            p_xmax = int(p_xmax * new_width)
            p_ymax = int(p_ymax * new_height)

            # Нарисуем найденный bb на фрейме
            cv2.rectangle(frame, (p_xmin, p_ymin), (p_xmax, p_ymax), (0, 0, 255), 2)
            # Запись видео
            out.write(frame)
            # Показываем фрейм
            cv2.imshow('Video ' + str((new_width, new_height)), frame)

            # Для выхода нажать Q
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    # Закрываем объекты capture
    cap.release()
    out.release()
    # Закрываем окна
    cv2.destroyAllWindows()

    # Имя и путь выходного файла сейчас не меняются
    return out_FILE
