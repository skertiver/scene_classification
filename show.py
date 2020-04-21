import numpy as np
import matplotlib.pyplot as plt

def show_confusion_mat(file_path):
    confusion_mat = np.load(file_path)
    print (confusion_mat.shape)

    normal_confusion_mat = confusion_mat / np.sum(confusion_mat,axis = 0)
    # plt.figure(figsize=(365,365))
    plt.imshow(normal_confusion_mat)
    plt.xlabel('predict label')
    plt.ylabel('groundtruth label')
    plt.figure(figsize=(365,365))
    plt.savefig('./confusion_mat.png',dpi=100)
    plt.close()

if __name__ == '__main__':
    show_confusion_mat('./validation_confusion_mat.npy')
