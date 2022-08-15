from datetime import date
import os
from urllib import response
import zipfile
import io
import random
import string

def renderFileName():
    # obj= date.today()
    # ran = ''.join(["%s" % random.randint(0, 8) for num in range(0, 9)])
    letters = string.ascii_lowercase
    ran = ''.join(random.choice(letters) for i in range(9))
    # return obj.strftime("%b-%d-%Y") + "_" + ran
    return ran

def getTypes(typeStr):
    return typeStr.split(',')

# def zipFiles(filenames):
#     zip_subdir = "archive"
#     zip_filename = "%s.zip" % zip_subdir

#     # Open StringIO to grab in-memory ZIP contents
#     s = io.StringIO.StringIO()
#     # The zip compressor
#     zf = zipfile.ZipFile(s, "w")

#     for fpath in filenames:
#         # Calculate path for file in zip
#         fdir, fname = os.path.split(fpath)
#         zip_path = os.path.join(zip_subdir, fname)

#         # Add file, at correct path
#         zf.write(fpath, zip_path)

#     # Must close zip for all contents to be written
#     zf.close()

#     # Grab ZIP file from in-memory, make response with correct MIME-type
#     resp = response(s.getvalue(), mimetype = "application/x-zip-compressed")
#     # ..and correct content-disposition
#     resp['Content-Disposition'] = 'attachment; filename=%s' % zip_filename

#     return resp