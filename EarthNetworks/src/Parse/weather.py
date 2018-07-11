import csv
import datetime
import glob
from Parse.utils import HourlyDataset

# example time format 5/13/2010 4:00:00 PM
time_format = "%m/%d/%Y %I:%M:%S %p"
start_time = (2011, 1, 1, 0)
end_time = (2014, 1, 1, 0)
output_path = "../../Data/parsed/CA"
filenames = glob.glob("../../Data/EarthNetworks/CAWeather/*.csv")

#fields = ["PressureSeaLevelMBar","PressureSeaLevelMBarRatePerHour","WindSpeedKph","WindSpeedKphAvg"]
#fields = "DewPointC,DewPointCRatePerHour,Humidity,HumidityRatePerHour,Light,LightRatePerHour,PressureSeaLevelMBar,PressureSeaLevelMBarRatePerHour,RainMillimetersDaily,RainMillimetersRatePerHour,RainMillimetersMonthly,RainMillimetersYearly,TemperatureC,TemperatureCRatePerHour,FeelsLike,WindSpeedKph,WindDirectionDegrees,WindSpeedKphAvg,WindDirectionDegreesAvg,WindGustKphHourly,WindGustTimeUtcHourly,WindGustDirectionDegreesHourly,WindGustKphDaily,WindGustTimeUtcDaily,WindGustDirectionDegreesDaily,HumidityHigh,HumidityHighUtc,HumidityLow,HumidityLowUtc,LightHigh,LightHighUtc,LightLow,LightLowUtc,PressureSeaLevelHighMBar,PressureSeaLevelHighUtc,PressureSeaLevelLowMBar,PressureSeaLevelLowUtc,RainRateMaxMmPerHour,RainRateMaxUtc,TemperatureHighC,TemperatureHighUtc,TemperatureLowC,TemperatureLowUtc".split(',')

# California:
fields = "Humidity, Humidity-QcDataDescriptor, HumidityRatePerHour, HumidityRatePerHour-QcDataDescriptor, Light, Light-QcDataDescriptor, LightRatePerHour, LightRatePerHour-QcDataDescriptor, PressureSeaLevelMBar, PressureSeaLevelMBar-QcDataDescriptor, PressureSeaLevelMBarRatePerHour, PressureSeaLevelMBarRatePerHour-QcDataDescriptor, RainMillimetersDaily, RainMillimetersDaily-QcDataDescriptor, RainMillimetersRatePerHour, RainMillimetersRatePerHour-QcDataDescriptor, RainMillimetersMonthly, RainMillimetersMonthly-QcDataDescriptor, RainMillimetersYearly, RainMillimetersYearly-QcDataDescriptor, TemperatureC, TemperatureC-QcDataDescriptor, TemperatureCRatePerHour, TemperatureCRatePerHour-QcDataDescriptor, WindSpeedKph, WindSpeedKph-QcDataDescriptor, WindDirectionDegrees, WindDirectionDegrees-QcDataDescriptor, WindSpeedKphAvg, WindSpeedKphAvg-QcDataDescriptor, WindDirectionDegreesAvg, WindDirectionDegreesAvg-QcDataDescriptor, WindGustKphHourly, WindGustKphHourly-QcDataDescriptor, WindGustTimeUtcHourly, WindGustDirectionDegreesHourly, WindGustDirectionDegreesHourly-QcDataDescriptor, WindGustKphDaily, WindGustKphDaily-QcDataDescriptor, WindGustTimeUtcDaily, WindGustDirectionDegreesDaily, WindGustDirectionDegreesDaily-QcDataDescriptor, HumidityHigh, HumidityHigh-QcDataDescriptor, HumidityHighUtc, HumidityLow, HumidityLow-QcDataDescriptor, HumidityLowUtc, LightHigh, LightHigh-QcDataDescriptor, LightHighUtc, LightLow, LightLow-QcDataDescriptor, LightLowUtc, PressureSeaLevelHighMBar, PressureSeaLevelHighMBar-QcDataDescriptor, PressureSeaLevelHighUtc, PressureSeaLevelLowMBar, PressureSeaLevelLowMBar-QcDataDescriptor, PressureSeaLevelLowUtc, RainRateMaxMmPerHour, RainRateMaxMmPerHour-QcDataDescriptor, RainRateMaxUtc, TemperatureHighC, TemperatureHighC-QcDataDescriptor, TemperatureHighUtc, TemperatureLowC, TemperatureLowC-QcDataDescriptor, TemperatureLowUtc"\
    .split(", ")
fields = [field for field in fields if "QcDataDescriptor" not in field]
print("Will parse: {}".format(fields))

for count, field in enumerate(fields):
    print("Parsing field ", count + 1, " of ", len(fields))
    files = [open(name, 'r') for name in filenames]
    try:
        with open(output_path + field + ".csv", 'w') as output_file:
            writer = csv.writer(output_file)
            hourly = HourlyDataset(start_time, files, ",", True)

            date, datas = next(hourly)
            while date < datetime.datetime(end_time[0], end_time[1], end_time[2], end_time[3]):
                writer.writerow(["" if data is None else data[field] for data in datas])
                date, datas = next(hourly)
    finally:
        for file in files:
            file.close()
