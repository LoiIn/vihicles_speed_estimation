from datetime import datetime
from random import randint

def renderFileName():
    obj = datetime.now()
    ran = ''.join(["%s" % randint(0, 8) for num in range(0, 9)])
    return obj.strftime("%y-%m-%d") + "_" + ran