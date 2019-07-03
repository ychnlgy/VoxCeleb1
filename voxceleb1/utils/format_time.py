import time

def format_time():
    tm = time.localtime()
    return "{}-{:0>2}-{:0>2}-{:0>2}:{:0>2}:{:0>2}".format(
        tm.tm_year,
        tm.tm_mon,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec
    )
