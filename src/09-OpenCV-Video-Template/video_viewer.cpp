#include "video_viewer.hpp"

VideoViewer::VideoViewer(VideoCapture& videoCapture)
{
	video = &videoCapture;
	*video >> mat;
	matViewer = new MatViewer("video", mat);
}


void VideoViewer::addToGUI(bool manualPlayback, bool imageControls)
{
	static bool loop = false;

	if (manualPlayback) {
		int frame = 0;
		frame = getCurrentFrame();
		if (ImGui::SliderInt("Video position", &frame, 0, numFrames())) {
			isPlaying = false;
			setCurrentFrame(frame);
		}

		ImGui::Checkbox("Loop", &loop);
		if (isPlaying) {
			if (ImGui::Button("Stop")) {
				isPlaying = false;
			}
		}
		else if (ImGui::Button("Play")) isPlaying = true;

		if (ImGui::Button("Next frame")) nextFrame();

	}

	if (isPlaying) {
		if (!nextFrame()) {
			if(!loop)isPlaying = false;
			setCurrentFrame(1);
		}
	}

	matViewer->addToGUI(imageControls);
}

int VideoViewer::getCurrentFrame()
{
	return (int)video->get(CAP_PROP_POS_FRAMES);
}

void VideoViewer::setCurrentFrame(int desiredFrame)
{
	if (desiredFrame == 0) {
		video->set(CAP_PROP_POS_FRAMES, 0);
		nextFrame();
		return;
	}
	desiredFrame--;
	video->set(CAP_PROP_POS_FRAMES, desiredFrame);
	nextFrame();
}

bool VideoViewer::nextFrame()
{
	if (!video->isOpened()) {
		cerr << "VideoCapture not opened" << endl;
		return false;
	}

	*video >> mat;

	if (mat.empty()) {
		cerr << "Mat is empty" << endl;
		return false;
	}

	matViewer->update();
	return true;
}

int VideoViewer::numFrames()
{
	return (int)video->get(CAP_PROP_FRAME_COUNT);
}
