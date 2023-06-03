# run inference experiments

import os

for i in range(2):
    dir_name = 'pair_'+str(i)
    print("processing dir "+str(i))
    try:
        os.system('python ../cal_column_similarity.py -p '+dir_name+' -m ../model/2022-04-12-12-06-32 -s one-to-one')
    except:
        print("An exception occurred")
    print()