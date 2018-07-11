import csv
import datetime
import glob
from Parse.utils import HourlyDataset

# example time format 5/13/2010 4:00:00 PM
time_format = "%m/%d/%Y %I:%M:%S %p"
start_time = (2011, 1, 1, 0)
end_time = (2014, 1, 1, 0)
output_path = "../../Data/EarthNetworks/CAWeather/"
filenames = glob.glob("../../Data/EarthNetworks/CAWeather/Unformatted/*.csv")

names = []
writers = {}

for count, filename in enumerate(filenames, 1):
    print("File {}/{}".format(count, len(filenames)))
    with open(filename, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        for row in reader:
            name = row[0]
            if name not in names:
                outfile = open(output_path + name + ".csv", 'w')
                writers[name] = csv.writer(outfile)
                writers[name].writerow(header)
                names.append(name)
            writers[name].writerow(row)
