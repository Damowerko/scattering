import csv
import os
import datetime
import itertools


in_path = "./"
out_path = "../../Data/parsed/"
#old_filename = "../../Data/Reliability/NyConsolidated2002-2010.csv"
new_filename = "../../Data/Reliability/CaConsolidated2011-2013.csv"
out_filename = "reliability_Ca.csv"

start_year = 2011
end_year = 2013

def parse_old(reader : csv.DictReader):
    return [] #TODO Implement parse_old

def parse_new(reader : csv.DictReader):
    header = next(reader)
    start_index = header.index("Event Date and Time")
    end_index = header.index("Restoration Date and Time")

    time_format = "%m/%d/%Y %I:%M %p" # eg. 05/29/2013  8:58 PM
    time_format_alternate = "%m/%d/%Y %H:%M"

    intervals = []
    for row in reader:
        try:
            start_time = datetime.datetime.strptime(row[start_index], time_format)
        except ValueError:
            start_time = datetime.datetime.strptime(row[start_index], time_format_alternate)

        try:
            end_time = datetime.datetime.strptime(row[end_index], time_format)
        except ValueError:
            end_time = datetime.datetime.strptime(row[end_index], time_format_alternate)

        def round_to_hour(time : datetime.datetime):
            time += datetime.timedelta(hours=round(time.hour / 60.0))
            time = time.replace(minute=0, second=0, microsecond=0)
            return time

        start_time = round_to_hour(start_time)
        end_time = round_to_hour(end_time)

        intervals.append((start_time, end_time))
    return intervals


if not os.path.exists(out_path):
    os.makedirs(out_path)
with open(in_path + new_filename) as new_f, open(out_path + out_filename, 'w') as out_f:  #open(in_path + old_filename) as old_f,
    # parse the old and new files
    intervals = parse_new(csv.reader(new_f)) #parse_old(csv.reader(old_f)) +

    # sort intervals
    intervals.sort(key=lambda pair: pair[0])

    # merge overlapping intervals
    i = 0
    while i < len(intervals)-1:
        first = intervals[i]
        second = intervals[i+1]
        # if intervals overlap then merge
        if first[1] > second[0]:
            intervals[i] = (first[0], max(first[1], second[1]))
            del intervals[i+1]
        i += 1

    def inside_intervals(time, intervals):
        for interval in intervals:
            if interval[0] < time < interval[1]:
                return True
            elif time < interval[0]:
                break
        return False

    fieldnames = ['Year', 'Month', 'Day', 'Hour', 'Fault']
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)  # csv writer for output value
    writer.writeheader()

    current_time = datetime.datetime(start_year, 1, 1, 0)
    while current_time.year <= end_year:
        values = dict()
        values['Year'] = current_time.year
        values['Month'] = current_time.month
        values['Day'] = current_time.day
        values['Hour'] = current_time.hour
        values['Fault'] = 0
        if inside_intervals(current_time, intervals):
            values['Fault'] = 1
        writer.writerow(values)

        current_time += datetime.timedelta(hours=1)



