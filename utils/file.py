import os

ROOT_PATH = './data'
FILE_EXTENSIONS = ('jpg', 'JPG', 'png', 'PNG', 'tif', 'gif', 'ppm')

def getSortedFilePaths(path):
    pathToFiles = os.path.join(ROOT_PATH, path)
    return sorted(
        [
            os.path.join(pathToFiles, fname)
            for fname in os.listdir(pathToFiles)
            if fname.endswith(FILE_EXTENSIONS) and not fname.startswith(".")
        ]
    )