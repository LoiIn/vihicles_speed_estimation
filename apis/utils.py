from datetime import date
from random import randint

def renderFileName():
    obj= date.today()
    ran = ''.join(["%s" % randint(0, 8) for num in range(0, 9)])
    return obj.strftime("%%b-%d-%Y") + "_" + ran