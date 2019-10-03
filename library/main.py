from library.PCA import PCA
from library.KPCA import KPCA
import configparser

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    method = config.get('ACTION', 'METHOD')
    path = config.get('ACTION', 'PATH')
    faces = config.getint('ACTION', 'FACES')
    v_size = config.getint('SIZES', 'V_SIZE')
    h_size = config.getint('SIZES', 'H_SIZE')
    eigenfaces_quantity = config.getint('DEFAULT', 'NUMBER_OF_EIGENFACES')
    images_dir = config.get('DEFAULT', 'IMAGE_DIR')
    images_quantity_per_person = config.getint('DEFAULT', 'IMAGES_PER_PERSON')
    people_quantity = config.getint('DEFAULT', 'NUMBER_OF_PEOPLE')
    training_n = config.getint('DEFAULT', 'TRAINING_NUMBER')
    test_n = config.getint('DEFAULT', 'TEST_NUMBER')

    # Train and Predict
    data = {
        'faces': faces,
        'path': path,
        'v_size': v_size,
        'h_size': h_size,
        'images_dir': images_dir,
        'images_quantity_per_person': images_quantity_per_person,
        'people_quantity': people_quantity,
        'training_n': training_n,
        'test_n': test_n,
        'eigenfaces_quantity': eigenfaces_quantity
    }

    print "KPCA\n"
    KPCA.predict(data)
    KPCA.test(data)

    print "PCA\n"
    PCA.test(data)
    PCA.predict(data)



if __name__ == "__main__":
    main()
