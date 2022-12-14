import time
import os


def tic():
    _start_time = time.time()
    return _start_time


def tac(_start_time):
    # Reference: https://newbedev.com/how-do-you-determine-a-processing-time-in-python
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    return (
        time.time(),
        "Time passed: {}hour:{}min:{}sec".format(t_hour, t_min, t_sec),
    )


def read_data(data_path):
    videos_path_list = []
    print("List of all directories in '% s':" % data_path)
    files = sorted(os.listdir(data_path))
    for video_name in files:
        # Se o arquivo for txt
        if video_name.endswith(".avi"):
            path = data_path + video_name
            videos_path_list.append(path)
            # print(path)

    return videos_path_list


def read_data_with_subdirectorys(data_path, ext='.avi'):
    videos_path_list = []
    print("List of all directories in '% s':" % data_path)

    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if name.endswith(ext):
                videos_path_list.append(os.path.join(path, name))

    return videos_path_list


def create_directory(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        #print("Directory ", dirName, " Created ")
    #else:
        #print("Directory ", dirName, " already exists")
    return dirName
