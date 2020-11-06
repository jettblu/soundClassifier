import librosa
import librosa.display as ld
from librosa import feature
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import *
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from plotMesh import plot_decision_boundaries
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import Normalizer
import seaborn as sns
from sklearn.svm import SVC
import math

# sampling rate of stored audio files
rate = 22050


def nextPow2(n):
    return math.ceil(math.log2(abs(n)))

def loadAlarmData():
    alarmData = np.load('alarm.npy', allow_pickle=True)
    fv = featurizeInput(alarmData)
    return fv


def loadBlenderData():
    blenderData = np.load('blender.npy', allow_pickle=True)
    fv = featurizeInput(blenderData)
    return fv


def loadMicrowaveData():
    microwaveData = np.load('microwave.npy', allow_pickle=True)
    fv = featurizeInput(microwaveData)
    return fv

def loadMusicData():
    musicData = np.load('music.npy', allow_pickle=True)
    fv = featurizeInput(musicData)
    return fv


def loadSilenceData():
    silenceData = np.load('silence.npy', allow_pickle=True)
    fv = featurizeInput(silenceData)
    return fv


def loadVacuumData():
    vacuumData = np.load('alarm.npy', allow_pickle=True)
    fv = featurizeInput(vacuumData)
    return fv


def preprocess(rawData):
    trimData, _ = librosa.effects.trim(rawData)
    return trimData


# returns feature vector for a sample
def getFeatureVector(rawSound):
    # L = len(rawSound)//6
    # n_fft = 2 ** nextPow2(L)
    # n_fft also used as window size
    n_fft = 2048
    hop = 512

    # features to consider
    fnList1 = [
        feature.chroma_stft,
        feature.spectral_centroid,
        feature.spectral_bandwidth,
        feature.spectral_rolloff,
        feature.melspectrogram,
    ]

    fnList2 = [
        feature.rms,
        feature.zero_crossing_rate,
        feature.spectral_flatness,
    ]

    # creates power spectogram
    D = np.abs(librosa.stft(rawSound, n_fft=n_fft, hop_length=hop))
    fnList3 = [
        np.max(D),
        np.std(D),
        np.mean(D),
        np.min(D)
    ]

    # median is used so as to mitigate effect of outliers
    featList1 = [np.median(funct(rawSound, rate)) for funct in fnList1]
    featList2 = [np.median(funct(rawSound))//len(rawSound) for funct in fnList2]
    # retrieve mfcc for audio data
    mfccs = librosa.feature.mfcc(rawSound, rate, n_mfcc=13, fmax=7000)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    # combine features into single list
    featList = featList1 + featList2 + fnList3
    for coeff in mfccs_scaled:
        featList.append(coeff)
    return featList


# receives array of raw sounds for particular class of sound as input
# returns features of that class
def featurizeInput(typeRawSounds):
    out = []
    for sample in typeRawSounds:
        sample = preprocess(sample)
        fv = getFeatureVector(sample)
        out.append(fv)
    out = np.array(out)
    return out


def plotWaves(raw_sounds):
    i = 1
    for f in raw_sounds:
        plt.subplot(len(raw_sounds),1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        # plt.title(f"{i}")
        i += 1
    plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=18)
    plt.show()


# create wave, HZ, and power spec charts for a single file
def visualizeIndividualFile(rawData):
    ld.waveplot(rawData, sr=rate)

    plt.show()

    trimData, _ = librosa.effects.trim(rawData)
    n_fft = 2048

    hop_length = 512
    D = np.abs(librosa.stft(trimData, n_fft=n_fft, hop_length=hop_length))
    DB = librosa.amplitude_to_db(D, ref=np.max)

    plt.plot(D)
    plt.show()

    librosa.display.specshow(DB, sr=rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.show()


# establish labels and combined data set of all sounds
def setLabelsAndData(useStored=False, store=True):
    # use stored features if specified
    if useStored:
        data = np.load('data.npy', allow_pickle=True)
        labels = np.load('labels.npy', allow_pickle=True)
        return data, labels
    # load raw audio data
    alarm = loadAlarmData()
    blender = loadBlenderData()
    microwave = loadMicrowaveData()
    music = loadMusicData()
    silence = loadSilenceData()
    vacuum = loadVacuumData()
    sounds = [alarm, blender, microwave, music, silence, vacuum]
    labels = []
    data = []
    # store features for quicker testing if specified
    for i, sound in enumerate(sounds):
        labels.extend([i]*len(sound))
        # add each sample from of each sound to collective data set
        for sample in sound:
            data.append(sample)
    if store:
        np.save('data.npy', np.array(data))
        np.save('labels.npy', np.array(labels))
    return np.array(data), np.array(labels)


# calculates the mean accuracy for a given classifier over a number of trials
def getAccuracy(classifier, data, labels):
    testScores = []
    # set up container for class level results of each classification trial
    cv = KFold(n_splits=10, random_state=65, shuffle=True)
    # make predictions
    for train_index, test_index in cv.split(data):
        dataTrain, dataTest, labelsTrain, labelsTest = data[train_index], data[test_index], labels[train_index], labels[test_index]
        classifier.fit(dataTrain, labelsTrain)
        # create confusion matrix
        testScores.append(classifier.score(dataTest, labelsTest))
    return np.mean(testScores)


# scales and reduces dimensionality of feature vectors
def normalizeFeatures(data, labels, visualize):
    # scales data
    max_abs_scaler = MaxAbsScaler()
    data = max_abs_scaler.fit_transform(data)
    # applies principle component analysis
    pca = decomposition.PCA(n_components=5)
    pca.fit(data)
    data = pca.transform(data)
    # visualizes scaled feature spread
    if visualize:
        for i in range(data.shape[1]):
            sns.kdeplot(data[:, i])
        plt.show()
    return data


# returns the accuracy for a series of classifiers
def classify(useStored=True, store=True):
    data, labels = setLabelsAndData(useStored=useStored, store=store)
    data = normalizeFeatures(data, labels, visualize=True)
    clfs = [RandomForestClassifier(n_estimators=300), KNeighborsClassifier(n_neighbors=10), SVC(kernel="rbf")]
    # initializes dictionary that will contain classifier as a key and accuracy as a value
    accuracies = dict()
    meshPlots(data, labels)
    knnTest(data, labels)
    # retrieves accuracy of each classifier
    for clf in clfs:
        accuracy = getAccuracy(clf, data, labels)
        accuracies[str(clf)] = accuracy

    print(f"\tFINAL ACCURACY\nAchieved Using KNeighborsClassifier(n_neighbors=2)\nMean Accuracy: "
          f"{getAccuracy(KNeighborsClassifier(n_neighbors=2), data, labels)}\n"
          f"You'll find accuracies produced from classifier tests below\n\t------------")

    return accuracies


def meshPlots(data, labels):
    # create mesh plots for first two features with given classifiers
    plt.figure()
    plt.title("random Forest")
    plot_decision_boundaries(data, labels, RandomForestClassifier, n_estimators=300)
    plt.show()
    plt.figure()
    plt.title("SVC")
    plot_decision_boundaries(data, labels, SVC, kernel="rbf")
    plt.show()
    plt.figure()
    plt.title("Nearest Neighbors")
    plot_decision_boundaries(data, labels, KNeighborsClassifier, n_neighbors=2)
    plt.show()


# illustrates KNN performance with varying n_neighbors values
def knnTest(data, labels):
    nNeighbors = []
    accuracyValues = []

    for i in range(1, 51):
        accuracy = getAccuracy(KNeighborsClassifier(n_neighbors=i), data, labels)
        accuracyValues.append(accuracy)
        nNeighbors.append(i)

    accuracyMax = max(accuracyValues)
    xPos = accuracyValues.index(accuracyMax) + 1

    plt.title(f"KNN Accuracy vs. n_neighbors; Optimal n_neighbors = {xPos}")
    plt.xlabel("n_neighbors")
    plt.ylabel('Model Accuracy')
    plt.plot(nNeighbors, accuracyValues)

    plt.show()


accuracies = classify(useStored=True, store=True)

for clf in accuracies:
    print(f"{clf} Mean Accuracy: {accuracies[clf]}")



