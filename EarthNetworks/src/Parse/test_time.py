import datetime

start = datetime.datetime(2011,1,1,0)
end = datetime.datetime(2014,1,1,0)

days = 0
while start < end:
    start = start + datetime.timedelta(hours=1)
    days += 1

print(days)