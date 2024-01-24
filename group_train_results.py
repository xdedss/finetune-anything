import re
import os
import glob
import sys

path = sys.argv[1]

# Define the regular expression pattern to match the desired format
pattern = r'iteration : (\d+), mIoU : (\d+(?:\.\d+)?), best_valid_mIoU : (\d+(?:\.\d+)?), time : (\d+)'

for file_path in glob.glob(os.path.join(path, '*.log')):
    # Open the text file for reading
    print(file_path)
    with open(file_path, 'r') as file:
        for line in file:
            # Search for the pattern in each line
            match = re.search(pattern, line)
            
            # If a match is found, extract the values
            if match:
                iteration = int(match.group(1))
                mIoU = float(match.group(2))
                best_valid_mIoU = float(match.group(3))
                time = int(match.group(4))
                
                if (iteration == 9999):
                    print(f'Iteration: {iteration}, mIoU: {mIoU}, Best Valid mIoU: {best_valid_mIoU}, Time: {time}')
