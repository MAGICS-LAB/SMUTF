import os
import shutil
import random
import subprocess




path = '/root/Python-Schema-Matching/Test Data/gurufocus'

for i in range(17):
    path = '/root/Python-Schema-Matching/Test Data/gurufocus' + str(i)

    for name in os.listdir(path):
        path_folder = os.path.join(path, name)
        
        
        try:
            if len(os.listdir(path_folder)) == 4:
                shutil.move(path_folder, '/root/Python-Schema-Matching/Test Data/gurufocus')
        except:
            pass
        
    



        


    