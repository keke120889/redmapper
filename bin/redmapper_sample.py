#!/usr/bin/env python

import redmapper
import getopt

def usage():
    print "Usage: redmapper_sample.py -c conffile"

if __name__=='__main__':
    try:
        opts,args = getopt.gnu_getopt(sys.argv[1:],"hc:")
    except getopt.GetoptError,err:
        print str(err)
        usage()
        sys.exit(2)

    conffile = None

    for o,a in opts:
        if o in '-h':
            usage()
            sys.exit()
        elif o in '-c':
            conffile = a
        else:
            print 'Illegal option:: '+o
            assert False,"Unhandled option"

    if conffile is None:
        usage()
        sys.exit()

    # this will call some code.
    redmapper.redmapper.blah(conffile)

    

