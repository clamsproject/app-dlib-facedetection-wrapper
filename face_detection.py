import cv2
import dlib
#import face_recognition

from clams.serve import ClamApp
from clams.serialize import *
from clams.vocab import AnnotationTypes
from clams.vocab import MediaTypes
from clams.restify import Restifier


class FaceDetection(ClamApp):

    def appmetadata(self):
        metadata = {"name": "Face Detection",
                    "description": "This tool applies OpenCV Face Detection to the entire video.",
                    "vendor": "Team CLAMS",
                    "requires": [MediaTypes.V],
                    "produces": [AnnotationTypes.FACE]}
        return metadata

    def sniff(self, mmif):
        # this mock-up method always returns true
        return True

    def annotate(self, mmif_json):
        mmif = Mmif(mmif_json)
        video_filename = mmif.get_medium_location(MediaTypes.V)
        face_output = self.run_FD(video_filename, mmif_json) #FD output is a list of (frame number, [(x1, y1, x2, y2),...]) pairs
        new_view = mmif.new_view()
        contain = new_view.new_contain(AnnotationTypes.FACE)
        contain.producer = self.__class__

        for int_id, (start_frame, box_list) in enumerate(face_output):
            annotation = new_view.new_annotation(int_id)
            annotation.start = str(start_frame)
            annotation.end = str(start_frame)  # since we're treating each frame individually for now, start and end are the same
            annotation.feature = {'faces':box_list}
            annotation.attype = AnnotationTypes.FACE

        for contain in new_view.contains.keys():
            mmif.contains.update({contain: new_view.id})
        return mmif

    @staticmethod
    def run_FD(video_filename, mmif): # mmif here will be used for filtering out frames/
        cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        sample_ratio = 30
        result = []
        def process_image(f):
            proc = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            proc = cv2.medianBlur(proc, 5)
            return proc

        cap = cv2.VideoCapture(video_filename)
        counter = 0

        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            if counter % sample_ratio == 0:
                processed_frame = process_image(f)
                res = cnn_face_detector(processed_frame)
                boxes = []
                for face in res:
                    boxes.append((face.rect.top(), face.rect.left(), face.rect.bottom(), face.rect.right()))
                if len(boxes) > 0:
                    result.append((counter, boxes))
            counter += 1
        return result

if __name__ == "__main__":
    fd_tool = FaceDetection()
    fd_service = Restifier(fd_tool)
    fd_service.run()

