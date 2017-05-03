import os
import pickle
from scipy import ndimage
import pandas as pd
import sqlalchemy

segmented_dir = "./segmented"
main_dir = "./progress"
image_dir = "./images"
source_dir_name = "source"
segment_file_name = "segment.pickle"
progress_file_name = "progress.pickle"

def get_project_directory(name):
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    project_path = os.path.join(main_dir, name)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    return project_path    


def project_exists(name):
    if not os.path.exists(main_dir):
        return False
    project_path = os.path.join(main_dir, name)
    if not os.path.exists(project_path):
        return False
    segment_file = os.path.join(project_path, segment_file_name)
    if not os.path.exists(segment_file):
        return False
    return True


def write_segment_file(name, data):
    project_path = get_project_directory(name)
    segment_file_path = os.path.join(project_path, segment_file_name)
    output = open(segment_file_path, 'ab+')
    pickle.dump(data, output)
    output.close()


def replace_segment_file(name, data):
    if project_exists(name):
        remove_segment_file(name)
    write_segment_file(name, data)


def remove_segment_file(name):
    project_path = get_project_directory(name)
    segment_file_path = os.path.join(project_path, segment_file_name)
    os.remove(segment_file_path)


def read_segment_file(name):
    project_path = get_project_directory(name)
    segment_file_path = os.path.join(project_path, "segment.pickle")
    if project_exists(name):
        output = open(segment_file_path, 'rb')
        data = pickle.load(output)
        output.close()
        return data
    else:
        raise IOError("Can't read project, it doesn't exist.")


def get_target_image(name):
    path = os.path.join(image_dir, name + ".jpg")
    img = ndimage.imread(path)
    return img


def get_source_image(index):
    path = os.path.join(image_dir, source_dir_name, "img_%i.jpg" % index)
    img = ndimage.imread(path)
    return img


def get_source_indices():
    img_num_list = []
    for src_img in os.listdir(os.path.join(image_dir, source_dir_name)):
        if "jpg" in src_img:
            img_name = os.path.splitext(src_img)[0]
            img_num = img_name.split("_")[1]
            img_num_list.append(int(img_num))

    return img_num_list

def get_saved_progress(project_name, patch):
    path = os.path.join(get_project_directory(project_name), progress_file_name)
    if os.path.exists(path):
        progress = pd.read_csv(path)
        return progress
    else:
        init = pd.Dataframe(columns=["state", "loss", "t"])
        return init


def add_progress(name, patch, state, loss, t):
    path = os.path.join(get_project_directory(name), progress_file_name)
    progress = get_saved_progress(name)
    current_t = progress.loc[patch]["t"]
    progress.loc[patch]["t"] = [state, loss, current_t + t]
    progress.to_csv(path)

if __name__ == "__main__":
    pass
