from library.PCA import PCA
from library.KPCA import KPCA
import configparser

def main():

	config = configparser.ConfigParser()
	config.read('config.ini')
	method = parser.get('ACTION', 'METHOD')
	
    # pca = PCA()
    # pca.test()
    #kpca = KPCA()
    #kpca.test()


if __name__ == "__main__":
    main()
