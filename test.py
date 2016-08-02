import lmdb
import os

class te:
    def __init__(self, a):
        aa = a

env = lmdb.Environment('./lm_db/', readonly=False, map_size=1048576 * 1024, metasync=False, sync=True, map_async=True)

with env.begin(write=True) as lmdb_txn:
    for i in xrange(3000):
        a = te(str(i))
        lmdb_txn.put(str(i), a)

env.sync(True)



with env.begin(write = True) as txn:
    cursor = txn.cursor()
    for i in xrange(3000):

        a = cursor.get(str(i))
        print(a)





