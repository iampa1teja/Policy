from __future__ import annotations 

from types import SimpleNamespace 

from ultralytics.trackers.byte_tracker import BYTETracker 
from ultralytics.trackers.bot_sort import BOTrack
from ultralytics.engine.results import Boxes 


class Tracker:
    """
        Wrapper of Ultralytics object trackers.
    """
    def __init__(self):
        self.trackers = {
            "bytetrack": BYTETracker(
                args=self._bytetrack_args()
            ),
            "botsort": BOTrack(
                args=self._botsort_args()
            ),
        }

    @staticmethod
    def _bytetrack_args():
        return SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.10,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.80,
            fuse_score=True,
        )

    @staticmethod
    def _botsort_args():
        return SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=0.25,
            track_low_thresh=0.10,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.80,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto",
        )

    def track(
        self, 
        detections, 
        image_shape, 
        tracker: str = "bytetrack", 
        img = None, 
    ): 
        """
        detections: Detection tensor produced by CNN.detect(). 
        """
        tracker = tracker.lower() 

        if tracker not in self.trackers:
            raise ValueError(
                f"Unknown tracker '{tracker}'. "
                f"Available trackers: "
                f"{list(self.trackers.keys())}"
            )

        results = Boxes(
            detections,
            orig_shape=image_shape,
        )

        return self.trackers[tracker].update(
            results,
            img=img,
        )

    def reset_tracker(
        self,
        tracker: str | None = None, 
    ):
        if tracker is None: 
            for tracking_algorithm in self.trackers.values(): 
                tracking_algorithm.reset() 
            return 

        tracker = tracker.lower()  

        if tracker not in self.trackers:
            raise ValueError(
                f"Unknown tracker '{tracker}'. "
                f"Available trackers: "
                f"{list(self.trackers.keys())}"
            )

        self.trackers[tracker].reset()