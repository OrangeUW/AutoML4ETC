from matplotlib import pyplot
import pandas
import numpy
import tensorflow.keras.callbacks
import seaborn
import tensorflow.keras.utils
import tensorflow.keras as keras
import sklearn.metrics as SM
from .datagen import DataGen

from .predefs import HEADER_SIZE, NUM_TYPES, TYPE_MAP, MASKED_VAL, TYPE_NAMES

def plot_history(history):
    pyplot.plot(history['sparse_categorical_accuracy'])
    pyplot.plot(history['val_sparse_categorical_accuracy'])
    pyplot.title('Model accuracy')
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Validation'], loc='upper left')
    pyplot.show()
    pyplot.figure(2)
    pyplot.plot(history['loss'])
    pyplot.plot(history['val_loss'])
    pyplot.title('Model loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Validation'], loc='upper left')
    pyplot.show()
    
    
def plot_history_binary(history):
    pyplot.plot(history['binary_accuracy'])
    pyplot.plot(history['val_binary_accuracy'])
    pyplot.title('Model accuracy')
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Validation'], loc='upper left')
    pyplot.show()
    pyplot.figure(2)
    pyplot.plot(history['loss'])
    pyplot.plot(history['val_loss'])
    pyplot.title('Model loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Validation'], loc='upper left')
    pyplot.show()
    
def unonehot(x):
    return x.argmax(axis=-1)

def gen_getlabels(gen: DataGen, ntypes: int):
    n = len(gen)
    y = [tensorflow.keras.utils.to_categorical(gen[i][1], num_classes=ntypes) for i in range(n)]
    y = numpy.concatenate(y)
    assert y.shape[1] == ntypes
    
    y = unonehot(y)
    return y

def confusion_normalise(mat):
    row_sums = mat.sum(axis=-1)
    return mat / row_sums[:, numpy.newaxis]

#def model_confusion(m, gen: DataGen, normalize=False, names=TYPE_NAMES, class_weight=None, \
#                    verbose=False, return_heatmap=False, report_only=False):
#    predicted = m.predict_generator(gen)
#    
#    # Get the actual labels
#    y_test = gen_getlabels(gen, len(names))
#    y_pred = unonehot(predicted)
#    
#    report = None
#    if class_weight:
#        weights = numpy.vectorize(lambda x:class_weight[x])(y_test)
#        if verbose:
#            print("Shapes: {},{},{}".format(y_test.shape, y_pred.shape, weights.shape))
#        report = SM.classification_report(y_test, y_pred, sample_weight=weights, target_names=names, digits=5, output_dict=report_only)
#        if not report_only:
#            print(report)
#            print("Accuracy: {}".format(SM.balanced_accuracy_score(y_test, y_pred, adjusted=True, sample_weight=weights)))
#    else:
#        report = SM.classification_report(y_test, y_pred, target_names=names, digits=5, output_dict=report_only)
#        if not report_only:
#            print(report)
#            print("Accuracy: {}".format(SM.accuracy_score(y_test, y_pred, normalize=False)))
#    if report_only:
#        return report
#
#    confm = SM.confusion_matrix(y_test, y_pred)
#    if normalize:
#        confm = confusion_normalise(confm)
#    df_cm = pandas.DataFrame(confm,
#                  index = names,
#                  columns = names)
#    if return_heatmap:
#        return df_cm
#
#    fig, ax = pyplot.subplots(figsize=(10,7))
#    seaborn.heatmap(df_cm, annot=True, ax=ax)
#    ax.set_xlabel("Predicted")
#    ax.set_ylabel("Actual")
#
#    fig.show()
#    
#    return df_cm
    
def model_confusion(model, gen: DataGen, normalize=False, names=TYPE_NAMES, class_weight=None, verbose=False, plot=True, pdf_name = None):

    predicted = model.predict_generator(gen)

    # Get the actual labels
    y_test = gen_getlabels(gen, len(names))
    y_pred = unonehot(predicted)

    if class_weight:
        weights = numpy.vectorize(lambda x:class_weight[x])(y_test)
        if verbose:
            print("Shapes: {},{},{}".format(y_test.shape, y_pred.shape, weights.shape))
        report = SM.classification_report(y_test, y_pred, sample_weight=weights, target_names=names, digits=5,
                                          labels=list(range(len(names))), output_dict=False)
        print(report)
        print("Accuracy: {}".format(SM.balanced_accuracy_score(y_test, y_pred, adjusted=True, sample_weight=weights)))
    else:
        report = SM.classification_report(y_test, y_pred, labels=list(range(len(names))), target_names=names, digits=5, output_dict=False)
        print(report)
        #print("Accuracy: {}".format(SM.accuracy_score(y_test, y_pred, normalize=False)))
        print("Accuracy: {}".format(SM.accuracy_score(y_test, y_pred, normalize=True)))
        accuracy = SM.accuracy_score(y_test, y_pred, normalize=True)
                                     
    confm = SM.confusion_matrix(y_test, y_pred, labels=list(range(len(names))))

    if normalize:
        confm = confusion_normalise(confm)

    df_cm = pandas.DataFrame(confm, index=names, columns=names)

    if plot:
        fig, ax = pyplot.subplots(figsize=(10, 7))
        seaborn.set(font_scale=1.4)
        seaborn.heatmap(df_cm, annot=True, ax=ax, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.show()
        if pdf_name:  
            fig.savefig('confm_{}.pdf'.format(pdf_name), bbox_inches='tight', pad_inches=0.1)

    return df_cm, accuracy

def model_confusion_binary(m, gen, class_weight=None):
    y_pred = m.predict_generator(gen, verbose=1)
    y_test = numpy.concatenate([gen[i][1] for i in range(len(gen))])
    
    if class_weight:
        weights = numpy.array([class_weight[x] for x in y_test])
        clr = SM.classification_report(y_test, numpy.round(y_pred), sample_weight=weights)
    else:
        clr = SM.classification_report(y_test, numpy.round(y_pred))
        
    print(clr)
    return clr
    
def model_save(m, name: str):
    m.save_weights('Models/weights.{}.h5'.format(name))
    m.save('Models/model.{}.h5'.format(name))
def model_load(name: str):
    return keras.models.load_model("Models/model.{}.h5".format(name))
    
def callback_csv(name: str, d=""):
    return keras.callbacks.CSVLogger("{}Models/history.{}.csv".format(d, name), append=True)
def callback_saver(name: str, d=""):
    return keras.callbacks.ModelCheckpoint(d + "Models/" + name + "/model.{epoch:02d}.h5")
def callback_saver_bestonly(name: str, d="", quantity='val_sparse_categorical_accuracy'):
    return keras.callbacks.ModelCheckpoint(d + "Models/" + name + "/model.best-{epoch:02d}.h5",\
                                           monitor=quantity, save_best_only=True, mode='max')
def callback_saver_periodic(name: str, d="", quantity='val_sparse_categorical_accuracy', period=10):
    return keras.callbacks.ModelCheckpoint(d + "Models/" + name + "/model.{epoch:02d}.h5", period=period)

def gen_getlabels_not_unonehot(gen, ntypes: int):
    n = len(gen)
    y = [tensorflow.keras.utils.to_categorical(gen[i][1], num_classes=ntypes) for i in range(n)]
    y = numpy.concatenate(y)
    assert y.shape[1] == ntypes
    
    return y

def top_k_accuracy_foreach_class(m, gen, names, k=1):
    y_test = gen_getlabels_not_unonehot(gen, len(names))
    y_pred = m.predict_generator(gen)
    
    y_test2 = y_test.argmax(axis=-1)
    top_k_accuracy = [0 for i in range(len(names))]
    class_counter = [0 for i in range(len(names))]
    
    top_allK_accuracies = []
    
    for j in range(k):
        y_pred2 = y_pred.argmax(axis=-1)

        for i in range(len(y_pred)):
            if y_test2[i] == y_pred2[i]:
                top_k_accuracy[y_test2[i]] += 1
            y_pred[i][y_pred2[i]] = 0
            if j == 0:
                class_counter[y_test2[i]] += 1
        val = numpy.array(top_k_accuracy) / numpy.array(class_counter)
        print("top "+str(j+1)+" accuracy is: " + str(list(numpy.around(val*100, 2))))
    
        top_allK_accuracies.append(val)
        
    return top_allK_accuracies

def top_allK_accuracies_plotter(top_allK_accuracies, names):
    import matplotlib.pyplot as plt
    
    X = numpy.arange(len(names))
    colors = ['r','b','g','y','k','m']

    for i in range(len(top_allK_accuracies)):
        Y = top_allK_accuracies[i]
        plt.bar(X + 0.12*(i), Y, color=colors[i], align="center", width = 0.12).set_label("K =" +str(i+1))
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Top K accuracy')
#     plt.xlabel(names)
    plt.show()
    
def top_k_accuracy_mean(top_allK_accuracies, logarithmic=True):
    top_allK_accuracies = numpy.array(top_allK_accuracies)
    k = len(top_allK_accuracies)
    
    mean = numpy.mean(top_allK_accuracies, axis=0)
    mean = numpy.around(mean*100, 2)
    weights = numpy.logspace(1, k+1, num=k, base=2) if logarithmic else numpy.arange(1, k+1)
    weights = numpy.flip(weights)
    print("logarithmic weights: {}".format(list(weights)))
    mean_weighted = numpy.average(top_allK_accuracies, axis=0, weights=weights)
    mean_weighted = numpy.around(mean_weighted*100, 2)
    print("Mean is: {} %".format(list(mean)))
    print("Weighted mean is: {} %".format(list(mean_weighted)))
    return [list(mean), list(mean_weighted)]

def top_k_full_analyze(m, gen, names, k=1, logarithmic=True):
    top_allK_accuracies = top_k_accuracy_foreach_class(m, gen, names, k)
    top_allK_accuracies_plotter(top_allK_accuracies, names)
    print(names)
    return top_k_accuracy_mean(top_allK_accuracies, logarithmic)
    