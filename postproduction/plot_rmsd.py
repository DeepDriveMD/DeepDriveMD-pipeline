#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    """
    Parse the command line arguments
    """
    describe = '''
Read the CSV file containing the latent space coordinates and the RMSD
values. Then visualize this data.'''
    example = '''
    '''
    parser = argparse.ArgumentParser(description=describe,
                                     epilog=example,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_file",help="The CSV file name")
    parser.add_argument("--image",help="The name of the image file")
    return parser.parse_args()

def plot_data(args):
    '''
    Load the CSV file and plot the data
    '''
    data = pd.read_csv(args.csv_file)
    ncol = data.shape[1]
    fig = plt.figure()
    if ncol == 3:
        ax = fig.add_subplot()
        x = data.iloc[:,0]
        y = data.iloc[:,1]
        c = data.iloc[:,2]
        img = ax.scatter(x,y,c=c,cmap='gist_rainbow')
    elif ncol == 4:
        ax = fig.add_subplot(projection='3d')
        x = data.iloc[:,0]
        y = data.iloc[:,1]
        z = data.iloc[:,2]
        c = data.iloc[:,3]
        img = ax.scatter(x,y,z,c=c,cmap='gist_rainbow')
    fig.colorbar(img)
    if args.image == None:
        fig.show()
        while not fig.waitforbuttonpress(timeout=-1):
            pass
        plt.close(fig)
    else:
        fig.savefig(args.image)
        plt.close(fig)

if __name__ == "__main__":
    args = parse_args()
    plot_data(args)