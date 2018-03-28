import csv
import datetime
import glob
from Parse.utils import HourlyDataset

# example time format 5/13/2010 4:00:00 PM
time_format = "%m/%d/%Y %I:%M:%S %p"
start_time = (2011, 1, 1, 0)
end_time = (2014, 1, 1, 0)
output_path = "../../Data/parsed/"
filenames = glob.glob("../../Data/EarthNetworks/NYCWeather/*.csv")

fields = ["PressureSeaLevelMBar","PressureSeaLevelMBarRatePerHour","WindSpeedKph","WindSpeedKphAvg"]
for count, field in enumerate(fields):
    files = [open(name, 'r') for name in filenames]
    try:
        with open(output_path + field + ".csv", 'w') as output_file:
            writer = csv.writer(output_file)
            hourly = HourlyDataset(start_time, files)

            date, datas = next(hourly)
            while date < datetime.datetime(end_time[0], end_time[1], end_time[2], end_time[3]):
                writer.writerow(["" if data is None else data[field] for data in datas])
                print("Parsing: ", date.isoformat(), "Field ", count+1, " of ", len(fields))
                date, datas = next(hourly)

    finally:
        for file in files:
            file.close()
