import sys
from face_detection import FaceDetection
from datetime import datetime

st = datetime.now()
ocr = FaceDetection()
a = open(sys.argv[1])
b = a.read()
c = ocr.annotate(b)
for i in c.views:
    a = i.__dict__
    print (a)
    c = a.get("contains")
    bd = a.get("annotations")
    for d in bd:
        print (d.__dict__)
print (datetime.now()-st)