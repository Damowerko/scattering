import numpy as np
import typing as t


def get_parsed_data(name, region):
    return np.genfromtxt("../../Data/parsed/" + region + "/" + name + ".csv", delimiter=",")


def get_avaiability(region):
    return get_parsed_data("availability", region)


def get_labels(region):
    return get_parsed_data("reliability", region)[1:, 4:5]


def get_field_data(field, region):
    return get_parsed_data(field, region)


def pick_greedy(region, n_stations=25):
    A = get_avaiability(region)
    M = A.shape[1]  # number of stations
    x = np.full((M,), False)
    for i in range(M+1):
        if np.sum(x) == n_stations:
            break
        max_stations = 0
        max_x = np.zeros((M,))
        for j in range(0, M):
            x_new = x.copy()  # type: np.ndarray # the previous station choice
            x_new[j] = True
            if np.sum(x_new) == np.sum(x) + 1:
                stations = np.sum(np.sum(A[:, x_new.astype(bool)], 1) == np.sum(x_new))
                if np.sum(stations) > np.sum(max_stations):
                    max_stations = stations
                    max_x = x_new
        x = max_x
    assert np.sum(x) == n_stations
    return x.reshape((M,1))


def import_data(fields: t.List[str], stations: np.ndarray, region: str) -> (np.ndarray, np.ndarray):
    # Earth networks data
    datas_raw = [get_field_data(field, region) for field in fields]  # read data for each field
    datas = [data_raw[:, stations.reshape(data_raw.shape[1],)] for data_raw in datas_raw]  # use selected stations
    data = np.stack(datas, 2)
    n_missing = np.sum(np.sum(np.isnan(data), 2), 1)  # the number of missing datapoints
    I = n_missing == 0  # only take data with no missing values
    data = data[I, :, :]
    labels = get_labels(region)[I, :]
    assert data.shape[0] == labels.shape[0]
    return data, labels


if __name__ == "__main__":
    stations = pick_greedy("NYC")
    data, labels = import_data(["TemperatureC", "PressureSeaLevelMBar"], stations)
    assert len(data.shape) == 3
    data, labels = import_data(["TemperatureC"], stations, "NYC")
    assert len(data.shape) == 3