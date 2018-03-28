from typing import Tuple
import datetime
import csv

# example time format 5/13/2010 4:00:00 PM
time_format = "%m/%d/%Y %I:%M:%S %p"


def round_to_hour(time: datetime.datetime):
    time += datetime.timedelta(hours=round(time.hour / 60.0))
    time = time.replace(minute=0, second=0, microsecond=0)
    return time


class HourlyDataset:
    def __init__(self, start: Tuple[int, int, int, int], files):
        """

        :param start: Start Time (year, month, day, hour)
        :param files: The EarthNetwork files to read
        """
        self.files = files
        self.stations = [HourlyStation(start, file) for file in files]

    def __iter__(self):
        return self

    def __next__(self):
        data = [next(station) for station in self.stations]
        times, data = zip(*data)
        return times[0], data


class HourlyStation:
    def __init__(self, start: Tuple[int, int, int, int], file):
        self.file = file
        self.reader = csv.DictReader(file, delimiter=',')
        self.scan_time = datetime.datetime(start[0], start[1], start[2], start[3])

        self.data_buffer = None
        self.time_buffer = None
        self.end_of_file = False
        self.update_buffer()

    def __iter__(self):
        return self

    def update_buffer(self):
        past_buffer = (self.time_buffer, self.data_buffer)
        try:
            self.data_buffer = next(self.reader)
            self.time_buffer = datetime.datetime.strptime(self.data_buffer["ObservationTimeUtc"], time_format)
            self.time_buffer = round_to_hour(self.time_buffer)  # enforce nearest hour
        except StopIteration:
            self.end_of_file = True
            self.data_buffer = None
            self.time_buffer = None
        return past_buffer

    def increment_scan_time(self):
        current_time = self.scan_time
        self.scan_time += datetime.timedelta(hours=1)
        return current_time

    def __next__(self) -> (datetime.datetime, dict):
        while True:  # update buffer until it catches up or reaches eof
            if self.end_of_file:  # if at end of file return scan time
                return self.increment_scan_time(), None
            if self.time_buffer >= self.scan_time:
                break
            self.update_buffer()

        # if time_buffer ahead then there is no data for the current time
        if self.time_buffer > self.scan_time:
            return self.increment_scan_time(), None

        # return data for current time
        if self.time_buffer == self.scan_time:
            time, data = self.update_buffer()
            return self.increment_scan_time(), data
