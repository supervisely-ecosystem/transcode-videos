import os
import json
import subprocess

import supervisely as sly


def _get_frame_count(video_path: str):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            video_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return int(result.stdout.strip())


def _get_frame_timestamps(video_path: str):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "json",
            video_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    data = json.loads(result.stdout)
    timestamps = []
    for frame in data.get("frames", []):
        if "pts_time" in frame:
            timestamps.append(float(frame["pts_time"]))
    return timestamps


def _find_vfr_frame_for_timestamp(cfr_ts: float, vfr_timestamps: list) -> int:
    left, right = 0, len(vfr_timestamps) - 1
    while left < right:
        mid = (left + right + 1) // 2
        if vfr_timestamps[mid] <= cfr_ts:
            left = mid
        else:
            right = mid - 1
    return left


def _remap_annotations(
    project_meta: sly.ProjectMeta, ann_path: str, old_video_path: str, new_video_path: str
):
    try:
        old_frame_count = _get_frame_count(old_video_path)
        new_frame_count = _get_frame_count(new_video_path)

        if old_frame_count == new_frame_count:
            return

        video_name = os.path.basename(old_video_path)
        sly.logger.info(
            f"Remapping annotations for {video_name}: {old_frame_count} VFR frames -> {new_frame_count} CFR frames"
        )

        old_video_ann = sly.VideoAnnotation.load_json_file(ann_path, project_meta)
        if len(old_video_ann.frames) == 0:
            new_video_ann = sly.VideoAnnotation(
                img_size=old_video_ann.img_size,
                frames_count=new_frame_count,
                objects=old_video_ann.objects,
                frames=sly.FrameCollection([]),
                tags=old_video_ann.tags,
            )
            sly.json.dump_json_file(new_video_ann.to_json(), ann_path, indent=2)
            sly.logger.info(f"No frames to remap, updated frames_count to {new_frame_count}")
            return

        vfr_frames_dict = {frame.index: frame for frame in old_video_ann.frames}
        sly.logger.info(f"Loading timestamps for {len(vfr_frames_dict)} annotated VFR frames")
        vfr_timestamps = _get_frame_timestamps(old_video_path)
        cfr_timestamps = _get_frame_timestamps(new_video_path)

        new_frames = []
        cfr_indices = list(range(new_frame_count))

        with sly.tqdm_sly("Remapping annotation frames", total=new_frame_count) as progress:
            for cfr_batch in sly.batched(cfr_indices, batch_size=10000):
                batch_frames = []

                for cfr_idx in cfr_batch:
                    cfr_ts = cfr_timestamps[cfr_idx]
                    vfr_idx = _find_vfr_frame_for_timestamp(cfr_ts, vfr_timestamps)
                    if vfr_idx not in vfr_frames_dict:
                        continue
                    vfr_frame = vfr_frames_dict[vfr_idx]
                    new_figures = [
                        sly.VideoFigure(
                            video_object=figure.video_object,
                            geometry=figure.geometry,
                            frame_index=cfr_idx,
                        )
                        for figure in vfr_frame.figures
                    ]

                    if new_figures:
                        batch_frames.append(sly.Frame(index=cfr_idx, figures=new_figures))

                new_frames.extend(batch_frames)
                progress.update(len(cfr_batch))

        new_frames_collection = sly.FrameCollection(new_frames)
        new_video_ann = sly.VideoAnnotation(
            img_size=old_video_ann.img_size,
            frames_count=new_frame_count,
            objects=old_video_ann.objects,
            frames=new_frames_collection,
            tags=old_video_ann.tags,
        )

        sly.json.dump_json_file(new_video_ann.to_json(), ann_path, indent=2)
        sly.logger.info(
            f"Successfully created new annotations: {len(new_frames)} CFR frames "
            f"from {len(vfr_frames_dict)} VFR frames"
        )

    except Exception as e:
        sly.logger.warning(f"Failed to remap annotations for {ann_path}: {e}", exc_info=True)
