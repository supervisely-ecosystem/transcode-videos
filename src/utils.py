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


def _build_vfr_to_cfr_map(vfr_timestamps: list, cfr_timestamps: list) -> dict:
    frame_map = {}

    for cfr_idx, cfr_ts in enumerate(cfr_timestamps):
        vfr_idx = 0
        for i, vfr_ts in enumerate(vfr_timestamps):
            if i + 1 < len(vfr_timestamps):
                next_vfr_ts = vfr_timestamps[i + 1]
                if vfr_ts <= cfr_ts < next_vfr_ts:
                    vfr_idx = i
                    break
            else:
                vfr_idx = i

        frame_map[cfr_idx] = vfr_idx

    return frame_map


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
        vfr_timestamps = _get_frame_timestamps(old_video_path)
        cfr_timestamps = _get_frame_timestamps(new_video_path)
        cfr_to_vfr_map = _build_vfr_to_cfr_map(vfr_timestamps, cfr_timestamps)

        old_video_ann = sly.VideoAnnotation.load_json_file(ann_path, project_meta)
        vfr_frames_dict = {}
        for frame in old_video_ann.frames:
            vfr_frames_dict[frame.index] = frame

        new_frames = []
        for cfr_idx in range(new_frame_count):
            if cfr_idx not in cfr_to_vfr_map:
                continue

            vfr_idx = cfr_to_vfr_map[cfr_idx]
            if vfr_idx not in vfr_frames_dict:
                continue

            vfr_frame = vfr_frames_dict[vfr_idx]
            new_figures = []
            for figure in vfr_frame.figures:
                new_figure = sly.VideoFigure(
                    video_object=figure.video_object,
                    geometry=figure.geometry,
                    frame_index=cfr_idx,
                )
                new_figures.append(new_figure)

            if new_figures:
                new_frame = sly.Frame(index=cfr_idx, figures=new_figures)
                new_frames.append(new_frame)

        new_frames_collection = sly.FrameCollection(new_frames)
        new_video_ann = sly.VideoAnnotation(
            img_size=old_video_ann.img_size,
            frames_count=new_frame_count,
            objects=old_video_ann.objects,
            frames=new_frames_collection,
            tags=old_video_ann.tags,
        )

        sly.json.dump_json_file(new_video_ann.to_json(), ann_path, indent=2)
        sly.logger.info(f"Successfully created new annotations with {len(new_frames)} frames")
    except Exception as e:
        sly.logger.warning(f"Failed to remap annotations for {ann_path}: {e}", exc_info=True)
