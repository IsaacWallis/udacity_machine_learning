import os
import pickle

segmented_dir = "./segmented/"
main_dir = "./progress"

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
    return True
    
def write_segment_file(name, data):
    project_path = get_project_directory(name)
    segfile_path = os.path.join(project_path, "segment.pickle")
    output = open(segfile_path, 'ab+')
    pickle.dump(data, output)
    output.close()

def read_segment_file(name):
    project_path = get_project_directory(name)
    segfile_path = os.path.join(project_path, "segment.pickle")
    output = open(segfile_path, 'rb')
    data = pickle.load(output)
    output.close()
    return data

if __name__ == "__main__":
    pass
