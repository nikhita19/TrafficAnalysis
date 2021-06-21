import numpy as np
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

from . import helpers
from .detector import Detector
from .tracker import Tracker

class GlobalTracker(object):
    def __init__(self,MAX_TRACKERS=150, max_age = 20):

        # print('start')
        self.det = Detector()

        self.max_age = max_age  # no.of consecutive unmatched detection before
                     # a track is deleted

        self.min_hits =1  # no. of consecutive matches needed to establish a track

        self.tracker_list =[] # list for trackers

        self.MAX_TRACKERS = MAX_TRACKERS
        # Modified list for tracker IDs.
        self.track_id_list = deque(list(range(0,self.MAX_TRACKERS)))


    def assign_detections_to_trackers(self,trackers, detections, iou_thrd = 0.3):
        '''
        From current list of trackers and new detections, output matched detections,
        unmatchted trackers, unmatched detections.
        '''
        # print('inside assign_detections_to_trackers')
        # print('tracker: ',trackers)
        # print('detections:  ',detections)


        IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
        for t,trk in enumerate(trackers):
            #trk = convert_to_cv2bbox(trk)
            for d,det in enumerate(detections):
             #   det = convert_to_cv2bbox(det)
                IOU_mat[t,d] = helpers.box_iou2(trk,det)

        # Produces matches
        # Solve the maximizing the sum of IOU assignment problem using the
        # Hungarian algorithm (also known as Munkres algorithm)

        matched_idx = linear_assignment(-IOU_mat)

        unmatched_trackers, unmatched_detections = [], []
        for t,trk in enumerate(trackers):
            if(t not in matched_idx[:,0]):
                unmatched_trackers.append(t)

        for d, det in enumerate(detections):
            if(d not in matched_idx[:,1]):
                unmatched_detections.append(d)

        matches = []

        # For creating trackers we consider any detection with an
        # overlap less than iou_thrd to signifiy the existence of
        # an untracked object

        # print('\nmatches before matching index: ',matches)

        # print('\nmatched_idx:  ',matched_idx)


        for m in matched_idx:
            if(IOU_mat[m[0],m[1]]<iou_thrd):
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        # print('last step of this function assign_detections_to_trackers which returns  ')
        # print('matches after mathced index : ',matches)
        # print('np.array(unmatched_detections):  ',np.array(unmatched_detections))
        # print('np.array(unmatched_trackers):  ',np.array(unmatched_trackers))
        # print('\n')

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers), IOU_mat


    def pipeline(self, boxes,scores,classes,img, iou_threshold = 0.3, FaceTracker = False,
                return_tracker_id = False):
        """Pipeline function for detection and tracking

        Args:
            boxes: Bounding boxes detected.
            scores: Scores for the bounding boxes.
            classes: Class index.
            img: Input image
            iou_threshold: Detection overlap threshold
            return_tracker_id: If enabled, will return the tracker id. Used for compatibility for older versions
            of our code.

        Returns:
            o_boxes: Tracked bounding boxes. Size (N)
            out_scores_arr: Corresponding scores. Size (N)
            out_classes_arr: Corresponding classes. Size (N)
            img: Output Image.

        """

        # print('\ninside the pipeline\n')
        if (not FaceTracker):
           z_box, idx_vec = self.det.get_localization(boxes,
                                          scores,
                                          classes,
                                    img)
        else:
           z_box, idx_vec = self.det.get_localization(boxes,
                                          scores,
                                          classes,
                                    img)
        x_box =[]

        # print('first step- self.tracker_list:  ',self.tracker_list)

        if len(self.tracker_list) > 0:
            for trk in self.tracker_list:
                x_box.append(trk.box)
        
        # print('x_box is created:  ',x_box)
        # print (' going for the matching the detection')


        matched, unmatched_dets, unmatched_trks, IOU_mat \
        = self.assign_detections_to_trackers(x_box, z_box, iou_thrd = iou_threshold)

       

        # Deal with matched detections
        if matched.size >0:
            # print("matching done now dealing with the matched detection")
            for trk_idx, det_idx in matched:
                z = z_box[det_idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk= self.tracker_list[trk_idx]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                x_box[trk_idx] = xx
                tmp_trk.box =xx
                tmp_trk.hits += 1
                # print('orig. index: ' ,det_idx)
                # print('Track index: ', trk_idx)


        # Deal with unmatched detections
        
        if len(unmatched_dets)>0:
            # print('if len(unmatched_dets)>0: --> Deal with unmatched detections')
            for idx in unmatched_dets:
                z = z_box[idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = Tracker() # Create a new tracker
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.id = self.track_id_list.popleft() # assign an ID for the tracker
                tmp_trk.class_labels = classes[idx_vec[idx]] # assign the corresponding class label
                tmp_trk.class_scores = scores[idx_vec[idx]] # assign the corresponding class score
                self.tracker_list.append(tmp_trk)
                x_box.append(xx)

        # Deal with unmatched tracks
        if len(unmatched_trks)>0:
            # print('if len(unmatched_trks)>0:  --> Deal with unmatched tracks')
            for trk_idx in unmatched_trks:
                tmp_trk = self.tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box =xx
                x_box[trk_idx] = xx


        # The list of tracks to be annotated
        # print('The list of tracks to be annotated')

        good_tracker_list =[]
        out_boxes = []
        out_scores = []
        out_classes = []
        out_trk_id = []
        for trk in self.tracker_list:
            if ((trk.hits >= self.min_hits) and (trk.no_losses <=self.max_age)):
                good_tracker_list.append(trk)
                # We need to rearrange the bounding box since
                # it currently puts it into a yolo format.
                # ymin, xmin, ymax, xmax
                bounding_boxes = trk.box
                out_boxes.append(bounding_boxes)
                out_scores.append(trk.class_scores)
                out_classes.append(trk.class_labels)
                out_trk_id.append(trk.id)
        # Book keeping
        deleted_tracks = filter(lambda x: x.no_losses >self.max_age, self.tracker_list)
        # print('deleted_tracks:  ',deleted_tracks)

        for trk in deleted_tracks:
                self.track_id_list.append(trk.id)

        self.tracker_list = [x for x in self.tracker_list if x.no_losses<=self.max_age]


        # Before we return the result, we need to convert to an
        # array, then normalize the bounding box values between 0 and 1.
        # Convert them to arrays
        out_boxes_arr = np.asarray(out_boxes)
        out_scores_arr = np.asarray(out_scores)
        out_classes_arr = np.asarray(out_classes)
        out_trk_id_list = list(out_trk_id)

        # Normalize the box values. Copy to new array to prevent overwriting the old one.
        if out_boxes_arr.size > 0: # Check to ensure array isn't empty.
            o_boxes = np.zeros(shape=(out_boxes_arr.shape[0],out_boxes_arr.shape[1]))
            o_boxes[:,0] = out_boxes_arr[:,0] / float(img.shape[0])
            o_boxes[:,2] = out_boxes_arr[:,2] / float(img.shape[0])
            o_boxes[:,1] = out_boxes_arr[:,1] / float(img.shape[1])
            o_boxes[:,3] = out_boxes_arr[:,3] / float(img.shape[1])
        else:
            o_boxes = np.asarray([]) # Return an empty array

     
        if return_tracker_id:
            return out_trk_id_list, o_boxes, out_scores_arr, out_classes_arr, img, IOU_mat
        else:
            return o_boxes, out_scores_arr, out_classes_arr, img
