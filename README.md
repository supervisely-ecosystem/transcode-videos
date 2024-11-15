<div align="center" markdown>
<img src="https://github.com/user-attachments/assets/0b6cdd36-21c1-4f3f-b396-e66ccc476052"/>  


# Transcode Videos

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/transcode-videos)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/transcode-videos)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/transcode-videos.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/transcode-videos.png)](https://supervisely.com)

</div>

## Overview

This app ensures consistent video playback across different software by normalizing and transcoding videos to a common format. Video files may sometimes display inconsistently across various applications due to differing codec support or handling methods. This app standardizes playback and frame display.
When run, the app creates a new project with the same structure as the original, but with all videos transcoded using x264 video codec, AAC audio codec, MP4 container format and constant frame rate, ensuring compatibility with most video players and editors.

## How To Use

**Step 1:** Run the app from the context menu of a video project or from the ecosystem page.

<img src="https://github.com/user-attachments/assets/5cb35377-036d-41ef-ac57-52040c4844bf" width="450px"/>

**Step 2:** Wait for the task to complete. The resulting project, containing transcoded videos, will be created in the same workspace.

<img src="https://github.com/user-attachments/assets/751dadaf-18b3-4fa1-99bc-865d9db897ba"/>
<img src="https://github.com/user-attachments/assets/b9228160-3b3f-4cdb-9f05-09b95121617e"/>
