import pandas as pd
import model as md
from request_data import pull_data
import time
import sys

def main():
    ### Atualizar dataset
    pull_data() #comentar com superusuario

    data = pd.read_csv('../data/data.csv')

    # print(data.info())

    Model = md.model(data, int(sys.argv[1]), int(sys.argv[2]))
    Model.build_model()

    return 0


if __name__ == '__main__':
    i = time.time()
    main()

    f = time.time()
    # print(f - i)
