import librosa
import librosa.display as ld
import numpy as np
import matplotlib.pyplot as plt
import os


file_name = "C:\\Users\\jett2\\OneDrive\\Documents\\carnegie mellon\\classes\\ml\\projects\\project 2\\data\\alarm\\ZOOM0048.wav"


# returns 1d list of sound type name and 2d list of file paths
def getFileNames():
    mainDir = "C:\\Users\\jett2\\OneDrive\\Documents\\carnegie mellon\\classes\\ml\\projects\\project 2\\data"
    folderNames = []
    fileNames = []
    for folder in os.listdir(mainDir):
        folderPaths = []
        folderNames.append(folder)
        subDir = mainDir + f"\\{folder}"
        for fileName in os.listdir(subDir):
            filePath = subDir + "\\" + fileName
            folderPaths.append(filePath)
        fileNames.append(folderPaths)
    return folderNames, fileNames


folderNames, fileNames = getFileNames()


def load_sound_files(filePaths):
    rawSounds = []
    for fp in filePaths:
        X,sr = librosa.load(fp)
        print(np.shape(X))
        rawSounds.append(X)
    return rawSounds


# load raw sound for each file of each folder
# return 2d list containing the above where each item is array
def getRawSound(fileNames):
    rawSounds = []
    for folder in fileNames:
        rawSoundData = load_sound_files(folder)
        rawSounds.append(rawSoundData)
    return rawSounds


def saveAsNpy(rawSoundsList):
    for i, folder in enumerate(rawSoundsList):
        # turns list of arrays into np array of arrays
        arr = np.array(folder)
        folderName = folderNames[i]
        np.save(f'{folderName}.npy', arr, dtype=object)


# data = np.load('alarm.npy', allow_pickle=True)
# print(np.shape(list(data)[0]))
#
# plot_waves(list(data))
