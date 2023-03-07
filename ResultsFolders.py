import os

class ResultsFolders:
    def __init__(self, rootFolder, separator):
        self._root = rootFolder
        self._separator = separator
        
    
    def createFolderIfDoesNotExist(self, folderPath):
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
    
    def irg(self):
        path = self._root + self._separator + "irg" + self._separator
        self.createFolderIfDoesNotExist(path)
        return path
    
    def createPathString(self, folderPath, fileName):
        path = folderPath + self._separator + fileName
        return path