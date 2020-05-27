import os
from environment import reports_directory, config_directory, stock_decisions_directory, graph_directory, tuning_directory, error_file



def check_directories():
    if not os.path.isdir(reports_directory):
        os.mkdir(reports_directory)
    if not os.path.isdir(config_directory):
        os.mkdir(config_directory)
    if not os.path.isdir(stock_decisions_directory):
        os.mkdir(stock_decisions_directory)
    if not os.path.isdir(graph_directory):
        os.mkdir(graph_directory)
    if not os.path.isdir(tuning_directory):
        os.mkdir(tuning_directory)


def deleteFiles(dirObject , dirPath):
    if dirObject.is_dir(follow_symlinks=False):
        name = os.fsdecode(dirObject.name)
        newDir = dirPath+"/"+name
        moreFiles = os.scandir(newDir)
        for file in moreFiles:
            if file.is_dir(follow_symlinks=False):
                deleteFiles(file, newDir)
                os.rmdir(newDir+"/"+os.fsdecode(file.name))
            else:
                os.remove(newDir+"/"+os.fsdecode(file.name))
        os.rmdir(newDir)
    else:
        os.remove(dirPath+"/"+os.fsdecode(dirObject.name))


def delete_files_in_folder(directory):
    try:
        files = os.scandir(directory)
        for file in files:
            deleteFiles(file, directory)
    except:
        f = open(error_file, 'a')
        f.write("problem with deleting files in folder: " + directory + "\n")
        f.write(sys.exc_info()[1] + '\n')
        f.close()






