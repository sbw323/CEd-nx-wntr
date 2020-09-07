import os
from glob import glob

def main():
    i = 0

    for filename in os.listdir("xyz"):
        dst ="Hostel" + str(i) + ".jpg"
        src ='xyz'+ filename
        dst ='xyz'+ dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1

leak_files = glob("/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/*.csv")
leak_files[0]
