#! /usr/bin/env python3

import sys
import os
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

def average_scores(scores, bin_size, cumulative = False):
    avg_scores = []
    for i in range(len(scores) // bin_size):
        fst = i * bin_size if not cumulative else 0
        lst = (i + 1) * bin_size
        lst = lst if lst < len(scores) else None
        avg_scores.append(np.mean(scores[fst:lst]))
    return avg_scores

def graph_scores(filename):
    scores = []
    with open(filename, "r") as file:
        for line in file:
            scores.append(tuple(map(lambda x:int(x),line.split(","))))
    print("Num Scores:", len(scores))
    print("Avg Scores:", np.mean(scores))

    bin_size = 1000

    averages = average_scores(scores, bin_size)
    print("Plotting Average")
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title("Average Episode Reward Over Each %d Episodes" % (bin_size,))
    plt.xlabel("Episodes")
    plt.ylabel("Binned Average of End of Episode Reward")
    x = [bin_size * ep for ep in range(len(averages))]
    y = averages
    plt.plot(x,y, 'r')
    plt.savefig("average-per-%d-episodes.pdf" % (bin_size,))
    plt.savefig("average-per-%d-episodes.png" % (bin_size,))
    plt.close()

    cumulative = average_scores(scores, bin_size, True)
    print("Plotting Cumulative Average")
    x = [bin_size * ep for ep in range(len(cumulative))]
    y = cumulative
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title("Cumulative Average Episode Reward At Each %d Episodes" % (bin_size,))
    plt.xlabel("Episodes")
    plt.ylabel("Cumlative Average of End of Episode Reward")
    plt.plot(x,y, 'r')
    plt.savefig("cumulative-per-%d-episodes.pdf" % (bin_size,))
    plt.savefig("cumulative-per-%d-episodes.png" % (bin_size,))
    plt.close()


def main():
    parser = ArgumentParser(prog = sys.argv[0], description = "Graphs the score of each episode")
    parser.add_argument("filename",
            default = None,
            help = "csv file of the score for each episode"
            )
    args = parser.parse_args()
    graph_scores(args.filename)
    return 0

if __name__ == "__main__":
    rtn = main()
    sys.exit(rtn)
