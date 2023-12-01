import sys
sys.path.append("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

import ntpath

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

