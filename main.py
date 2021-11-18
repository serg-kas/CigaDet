"""
Демонстрация поиска сигареты на фото/видео
При запуске обрабатывает все файлы из папки source_files
Результат помещает в папку out_files, добавляя к имени "out_"
Если файл с таким названием уже обрабатывался, то его не трогает.
Видео во время обработки отображается.
"""
# Модуль с функицями
import cigadet

import os
import warnings
warnings.filterwarnings("ignore")

# Путь к модели
model_PATH = 'best-4-model-inception.h5'
# Папки
source_PATH = 'source_files'
out_PATH = 'out_files'
# Допустимые форматы
img_type_list = ['.jpg', '.jpeg', '.png']
vid_type_list = ['.mp4', '.avi']
#
if __name__ == '__main__':
    # Создадим папки для файлов, если их нет
    if not (source_PATH in os.listdir('.')):
        os.mkdir(source_PATH)
    if not (out_PATH in os.listdir('.')):
        os.mkdir(out_PATH)

    # В папке должен быть файл модели
    assert model_PATH in os.listdir('.'), 'В папке программы должен быть файл модели'

    # Создадим список файлов для обработки
    source_files = sorted(os.listdir(source_PATH))
    out_files = sorted(os.listdir(out_PATH))
    # Раздельные списки для картинок и видео
    img_files = []
    vid_files = []
    for f in source_files:
        filename, file_extension = os.path.splitext(f)
        # print(f,filename,file_extension)
        if not (('out_'+f) in out_files):
            if file_extension in img_type_list:
                img_files.append(f)
            if file_extension in vid_type_list:
                vid_files.append(f)

    # Получаем модель
    # model = cigadet.get_model(model_PATH, '')
    model = cigadet.get_model(model_PATH, '/cpu:0')

    # Обрабатываем картинки
    for img in img_files:
        # полные пути к файлам
        img_FILE = source_PATH + '/' + img
        out_FILE = out_PATH + '/' + 'out_' + img
        # Вызов функции предикта
        _ = cigadet.img_predict(model, img_FILE, out_FILE)

    # Обрабатываем видео
    for vid in vid_files:
        # полные пути к файлам
        vid_FILE = source_PATH + '/' + vid
        out_FILE = out_PATH + '/' + 'out_' + vid
        # Вызов функции предикта
        _ = cigadet.vid_predict(model, vid_FILE, out_FILE)

    # Сообщаем что обработали
    if len(img_files) == 0:
        print('Нет картинок для обработки.')
    else:
        print('Обработали {0} картинок.'.format(len(img_files)))
    if len(vid_files) == 0:
        print('Нет видео для обработки.')
    else:
        print('Обработали {0} видео.'.format(len(vid_files)))


