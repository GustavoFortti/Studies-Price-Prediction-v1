import pandas as pd
import numpy as np
import time

def main():
   
    f1 = open('filter_acc.txt', 'r')

    df = pd.DataFrame(f1.readlines())
    df = pd.DataFrame([(float(str(i).replace('n', '')[2:-3])) for i in np.array(df)])
    print(df.mean())
    print(len(df))

    return 0


if __name__ == '__main__':
    main()