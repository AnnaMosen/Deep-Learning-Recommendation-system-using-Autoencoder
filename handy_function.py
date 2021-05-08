def print_current_time(output = ''):
    import datetime
    import pytz
    current_time = datetime.datetime.now(pytz.timezone('Israel'))
    if output == '':
        print("The current time is: ")
    else:
        print(output)
    print(current_time)


def move_x_and_y_cpu( x, y):
        x = x.cpu()
        y = y.cpu()
        return (x,y)