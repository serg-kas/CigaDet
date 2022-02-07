# CigaDet

Cigarette detector in picture/frame as bounding box regression. 
We do not determine the presence of an object in the picture (that is, we do not perform object detection).
The object is always present and we only localize its bounding box.

Файл весов модели надо положить в папку программы.
(best-4-model-inception.h5 на google drive, размер ок.2Gb)

При первом запуске программа создаст если их нет папки out_files и source_files.

В папку source_files помещаем файлы фото или видео которые хотим обработать,
запускаем скрипт и в папке out_files получаем результат.

Обработанный файл в папке out_files имеет имя out_<имя исходного файла>
Повторно при следующих запусках один и тот же файл не обрабатывается 
(пока не удалить обработанный файл out_<имя исходного файла>)

Фото обрабатываются "молча".
Видео при обработке проигрывается.

По умолчанию предикт принудительно идет на процессоре.
Для предикта на устройстве по умолчанию надо в main.py заменить 
вызов функции model = cigadet.get_model(model_PATH, '/cpu:0')
на model = cigadet.get_model(model_PATH, '')
