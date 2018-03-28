from math import radians, cos, sin, asin, sqrt
import csv
import glob

def haversine(gps1, gps2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    (lon1, lat1) = gps1
    (lon2, lat2) = gps2

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


output_path = "../../Data/parsed/"
filenames = glob.glob("../../Data/EarthNetworks/NYCWeather/*.csv")
files = [open(name, 'r') for name in filenames]
try:
    with open(output_path + "distance.csv", 'w') as output_file:
        writer = csv.writer(output_file)
        readers = [csv.DictReader(file) for file in files]

        gps = []
        for reader in readers:
            try:
                data = next(reader)
                gps += [(float(data['Latitude']), float(data['Longitude']))]
            except StopIteration:
                gps += [None]

        for gps1 in gps:
            distance = []
            for gps2 in gps:
                if gps1 is None or gps2 is None:
                    distance += [""]
                else:
                    distance += [haversine(gps1, gps2)]
            writer.writerow(distance)
finally:
    for file in files:
        file.close()