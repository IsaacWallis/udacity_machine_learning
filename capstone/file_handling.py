import os
import pickle
from scipy import ndimage

segmented_dir = "./segmented"
main_dir = "./progress"
segment_file_name = "segment.pickle"
image_dir = "./images"
source_dir_name = "source"


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

if __name__ == "__main__":
    pass
