from library.PCA import PCA
from library.KPCA import KPCA
import configparser
from datetime import datetime

def main():
    config = configparser.ConfigParser()
    config.read('custom_images_config_example.ini')
    method = config.get('ACTION', 'METHOD')
    query = config.get('ACTION', 'QUERY')

    start = datetime.now()

    data = {
        'faces': config.getint('ACTION', 'FACES'),
        'path': config.get('ACTION', 'PATH'),
        'v_size': config.getint('SIZES', 'V_SIZE'),
        'h_size': config.getint('SIZES', 'H_SIZE'),
        'images_dir': config.get('DEFAULT', 'IMAGE_DIR'),
        'images_quantity_per_person': config.getint('DEFAULT', 'IMAGES_PER_PERSON'),
        'people_quantity': config.getint('DEFAULT', 'NUMBER_OF_PEOPLE'),
        'training_n': config.getint('DEFAULT', 'TRAINING_NUMBER'),
        'test_n': config.getint('DEFAULT', 'TEST_NUMBER'),
        'eigenfaces_quantity': config.getint('DEFAULT', 'NUMBER_OF_EIGENFACES')
    }

    if method == 'KPCA':
        print "Running KPCA\n"
        if query == 'PREDICT':
            print "Predicting ..."
            KPCA.predict(data)
        elif query == 'TEST':
            print "Testing ..."
            KPCA.test(data)
        else:
            print 'Bad Query'
    elif method == 'PCA':
        print "Running PCA\n"
        if query == 'PREDICT':
            print "Predicting ..."
            PCA.predict(data)
        elif query == 'TEST':
            print "Testing ..."
            PCA.test(data)
        else:
            print 'Bad Query'
    else:
        print 'Bad Query'

    duration = datetime.now() - start
    print ("Duration: {0}".format(duration.seconds))

    exit(0)


if __name__ == "__main__":
    main()
