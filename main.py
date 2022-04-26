import numpy as np
import matplotlib.pyplot as plt

from mnistClf import FashionClassifier
from apacheUse import ADealer
from googleUse import GDealer
from testData import TestData


def launch_clf(train=False, data_path='fashion', model_path='cnn_fashion_model2.ckpt'):
    # sets up a classifier

    fclf = FashionClassifier(train, data_path, model_path)
    final_model = fclf.model
    print('\nclf_launched')
    return final_model


def send_messages(pictures, dealer, model):
    # deals with Google and Apache

    messages = []
    for picture in pictures:
        message = picture.reshape((1, 28, 28, 1)).astype('float32')
        messages.append(message)
    dealer.write(messages)

    dealer.read_write_orders(model=model)

    read_messages = dealer.read()

    output = []
    for message in read_messages:
        if message.shape == (794,):
            received = np.split(message, [784])
            y = np.argmax(received[-1])
            x = received[0].reshape((28, 28))
            output.append((x, y))
    return output


def what_class_is_this(y):
    # reminds me what classifier results signify

    classes = {'0': "t-shirt/top",
               '1': "trousers",
               '2': "pullover",
               '3': "dress",
               '4': "coat",
               '5': "sandal,",
               '6': "shirt",
               '7': "sneaker",
               '8': "bag",
               '9': "ankle boot"}
    return classes[str(y)]


if __name__ == '__main__':
    # connects to user

    print("I know, it's not perfect, but this is my version of the solution.\n")

    # try and figure out clf params
    use_my_clf = str(input("If you want to use my classifier, write anything. " +
                           "If you want to use other classifier, write n.\n"))

    if use_my_clf != 'n':
        model = launch_clf()

    else:
        load_clf = str(input("If you want to load the classifier, write the path. " +
                             "If you want to train the classifier, write n.\n"))
        if load_clf == 'n':
            train_clf = str(input("\nIf you want to train the classifier, write path to the folder with data.\n"))
            model = launch_clf(train=True, data_path=train_clf)

        else:
            model = launch_clf(model_path=load_clf)

    # try and figure out dealer params
    use_apache = str(input("If you want to use Apache Kafka, write y. " +
                           "If you want to use Google Sub/Pub, write anything else.\n"))

    if use_apache == 'y':
        standard = str(input("If you want to use standard settings, write y. " +
                             "If you want to use your own server, write your server.\n"))
        if standard == 'y':
            dealer = ADealer()
        else:
            topic = str(input("Write your topic\n"))
            try:
                dealer = ADealer(host=standard, topic=topic)
            except Exception:
                print('Something went wrong, launching standard dealer.')
                dealer = ADealer()
    else:
        standard = str(input("If you want to use standard settings, write y. " +
                             "If you want to use your own server, write the path to your server key.\n"))
        if standard == 'y':
            dealer = GDealer()
        else:
            project = str(input("Write your project name\n"))
            topic = str(input("Write your topic\n"))
            try:
                dealer = GDealer(e_path=standard, project_id=project, topic_id=topic)
            except Exception:
                print('Something went wrong, launching standard dealer.')
                dealer = GDealer()

    # use something to test the machine
    use_data = str(input("If you want to use MNIST test data, write y. " +
                         "If you want to use your own data, write the path to it.\n"))

    if use_data == 'y':
        test_data = TestData().X_test
    else:
        test_data = TestData(path=use_data).X_test

    more = 'yes'
    while more != 'stop':
        request = 'yes'
        pictures = []
        while request != 'stop':
            request = str(input("Write picture number or 'stop' to stop.\n"))
            if request != 'stop':
                try:
                    pic_num = int(request)
                    normal_num = pic_num % test_data.shape[0]
                    pictures.append(test_data[normal_num])
                except Exception:
                    continue

        if pictures:
            output = send_messages(pictures=pictures, dealer=dealer, model=model)

        for o in output:
            print(what_class_is_this(o[-1]))
            plt.imshow(o[0], cmap=plt.cm.binary)
            plt.show()

        more = str(input("If you don't want to do it again, type 'stop'.\n"))
