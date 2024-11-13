import os
import shutil
import subprocess

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

project_id = sly.env.project_id(raise_not_found=True)
PROJECT_DIR = "project"
project_info = api.project.get_info_by_id(project_id, raise_error=True)


def _transcode(path: str, video_codec: str = "libx264", audio_codec: str = "aac"):
    output_path = path + "_transcoded.mp4"
    pcs = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            path,
            "-c:v",
            f"{video_codec}",
            "-c:a",
            f"{audio_codec}",
            "-vsync",
            "cfr",
            output_path,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if pcs.returncode != 0:
        raise RuntimeError(pcs.stderr)
    return output_path


# download project
sly.download_async(
    api,
    project_id,
    PROJECT_DIR,
)

# transcode videos
project = sly.VideoProject(PROJECT_DIR, sly.OpenMode.READ)
with sly.tqdm_sly(message="Transcoding videos...", total=project.total_items) as progress:
    for dataset in project:
        dataset: sly.VideoDataset
        for _, video_path, ann_path in dataset.items():
            try:
                output_path = _transcode(video_path)
            except Exception:
                sly.logger.warning(
                    "Failed to transcode video: %s. It will be skipped.", video_path, exc_info=True
                )
            else:
                result_path = video_path if video_path.endswith(".mp4") else video_path + ".mp4"
                # rename annotation file
                shutil.move(ann_path, ann_path.replace(video_path, result_path))
                # rename transcoded video file
                shutil.move(output_path, result_path)
                # delete original video file if needed
                if video_path != result_path:
                    os.remove(video_path)
            finally:
                progress.update(1)

# upload transcoded project
output_project_id, _ = project.upload(
    PROJECT_DIR,
    api,
    workspace_id=project_info.workspace_id,
    project_name=project_info.name + "_transcoded",
)


try:
    api.app.workflow.add_input_project(project_id)
    api.app.workflow.add_output_project(output_project_id)
except Exception:
    sly.logger.warning("Unable to set workflows for the task.", exc_info=True)
