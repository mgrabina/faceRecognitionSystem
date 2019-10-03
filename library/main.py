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
    # PCA.test()
    PCA.predict(path, data={
        'faces': faces,
        'path': path,
        'v_size': v_size,
        'h_size': h_size
    })


if __name__ == "__main__":
    main()
