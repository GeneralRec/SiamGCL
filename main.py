from SELFRec import SELFRec
from util.conf import ModelConf

if __name__ == '__main__':
    for i in range(1):
        import time
        s = time.time()

        model = 'SiamGCL'
        conf = ModelConf( model + '.conf')
        rec = SELFRec(conf)
        rec.execute()

        e = time.time()
        print("Running time: %f s" % (e - s))
