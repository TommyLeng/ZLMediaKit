/*
 * Copyright (c) 2016-present The ZLMediaKit project authors. All Rights Reserved.
 *
 * This file is part of ZLMediaKit(https://github.com/ZLMediaKit/ZLMediaKit).
 *
 * Use of this source code is governed by MIT-like license that can be found in the
 * LICENSE file in the root of the source tree. All contributing project authors
 * may be found in the AUTHORS file in the root of the source tree.
 */

// Regression test for the zero-connection snapshot path (getSnap internal GOP cache).
//
// FFmpegSnap::makeSnap(async=false, "rtmp://127.0.0.1/...") no longer spawns an
// external FFmpeg process for local streams; instead it drains the in-memory GOP
// cache of the already-running MediaSource and decodes a frame directly. This test
// exercises the three core-library pieces that path relies on, end to end:
//
//   1. MultiMediaSourceMuxer::flushGop()  - new public API exposing the GOP cache
//   2. FFmpegDecoder::flush()             - merger-then-avcodec drain fix, so a key
//                                           frame buffered only in the FrameMerger is
//                                           still decoded
//   3. FFmpegUtils::saveFrame()           - decoded frame -> JPEG on disk
//
// It loads a tiny MP4 fixture into a local MediaSource, waits until its GOP cache is
// populated, then reproduces makeSnapInternal()'s decode logic and asserts a valid
// JPEG is produced.
//
// Requires ENABLE_FFMPEG + ENABLE_MP4. The GOP cache itself is created under
// ENABLE_RTPPROXY (on by default); if it is disabled the test reports that and fails.
//
// Usage: test_getsnap [path/to/fixture.mp4]
//   Default fixture: <exeDir>/test_snap_fixture.mp4 (copied next to the binary by CMake)
//
// The bundled fixture is a 5s 640x360 H.264 clip with a key frame every second.
// To regenerate it:
//   ffmpeg -f lavfi -i color=c=red:size=640x360:rate=30:duration=1 \
//          -f lavfi -i color=c=green:size=640x360:rate=30:duration=1 \
//          -f lavfi -i color=c=blue:size=640x360:rate=30:duration=1 \
//          -filter_complex "[0][1][2]concat=n=3:v=1:a=0[v]" -map "[v]" \
//          -c:v libx264 -profile:v baseline -pix_fmt yuv420p -g 30 -keyint_min 30 \
//          -sc_threshold 0 tests/test_snap_fixture.mp4

#if defined(ENABLE_FFMPEG) && defined(ENABLE_MP4)

#include <unistd.h>
#include <vector>
#include <iostream>
#include "Util/File.h"
#include "Util/logger.h"
#include "Util/util.h"
#include "Util/TimeTicker.h"
#include "Common/config.h"
#include "Common/MediaSource.h"
#include "Common/MultiMediaSourceMuxer.h"
#include "Poller/EventPoller.h"
#include "Record/MP4Reader.h"
#include "Codec/Transcode.h"
#include "Extension/Frame.h"
#include "Extension/Track.h"

using namespace std;
using namespace toolkit;
using namespace mediakit;

namespace {

// Reproduces makeSnapInternal()'s decode stage: feed the cached GOP frames through a
// decoder and save the first decoded frame as a JPEG. Returns true on success.
bool decodeGopToJpeg(const VideoTrack::Ptr &track, const vector<Frame::Ptr> &gop, const string &save_path, string &err) {
    bool done = false;
    try {
        auto decoder = make_shared<FFmpegDecoder>(track);
        decoder->setOnDecode([&](const FFmpegFrame::Ptr &frame) {
            if (done) {
                return;
            }
            done = true;
            auto ret = FFmpegUtils::saveFrame(frame, save_path.data());
            if (!std::get<0>(ret)) {
                err = std::get<1>(ret);
            }
        });

        for (auto &f : gop) {
            if (done) {
                break;
            }
            decoder->inputFrame(f, false, false);
        }
        // The flush() fix: pushes a key frame still buffered in the FrameMerger into
        // avcodec, then drains avcodec's internal output buffer.
        if (!done) {
            decoder->flush();
        }
    } catch (const exception &e) {
        err = string("decode threw: ") + e.what();
        return false;
    }
    return done;
}

// A JPEG always starts with the SOI marker 0xFF 0xD8.
bool isJpeg(const string &path) {
    auto data = File::loadFile(path.data());
    return data.size() >= 2 && (uint8_t)data[0] == 0xFF && (uint8_t)data[1] == 0xD8;
}

} // namespace

int main(int argc, char *argv[]) {
    Logger::Instance().add(std::make_shared<ConsoleChannel>());
    Logger::Instance().setWriter(std::make_shared<AsyncLogWriter>());

    const string fixture = argc > 1 ? argv[1] : (exeDir() + "test_snap_fixture.mp4");
    if (!File::fileExist(fixture.data())) {
        ErrorL << "fixture not found: " << fixture << " (pass an MP4 path as argv[1])";
        return 1;
    }
    const string save_path = exeDir() + "test_snap_output.jpg";
    File::delete_file(save_path.data());

    // Register an eager local stream so MediaSource::find() can locate it without a
    // reader attached (rtmp_demand = false). Mirrors a real pushed/pulled local stream.
    ProtocolOption option;
    option.enable_rtmp = true;
    option.rtmp_demand = false;
    option.enable_audio = false;

    auto poller = EventPollerPool::Instance().getPoller();
    auto tuple = MediaTuple{DEFAULT_VHOST, "live", "snap_test", ""};
    auto reader = std::make_shared<MP4Reader>(tuple, fixture, option, poller);
    // ref_self = true keeps the reader alive for the duration; file_repeat = true loops
    // the short fixture so the stream stays up while we poll.
    reader->startReadMP4(100, true, true);

    // Poll until the stream is registered, its video track is ready, and the GOP cache
    // has buffered at least one frame.
    MediaSource::Ptr src;
    MultiMediaSourceMuxer::Ptr muxer;
    VideoTrack::Ptr track;
    vector<Frame::Ptr> gop;
    Ticker ticker;
    while (ticker.elapsedTime() < 15000) {
        usleep(100 * 1000);

        src = MediaSource::find(DEFAULT_VHOST, "live", "snap_test", false);
        if (!src) {
            continue;
        }
        muxer = src->getMuxer();
        if (!muxer) {
            continue;
        }
        track.reset();
        for (auto &t : muxer->getTracks(true)) {
            if (t->getTrackType() == TrackVideo) {
                track = dynamic_pointer_cast<VideoTrack>(t);
                break;
            }
        }
        if (!track) {
            continue;
        }
        gop.clear();
        muxer->flushGop([&](const Frame::Ptr &frame) {
            if (frame->getTrackType() == TrackVideo) {
                gop.push_back(frame);
            }
        });
        if (!gop.empty()) {
            break;
        }
    }

    if (!src) {
        ErrorL << "FAIL: stream never registered (live/snap_test)";
        return 1;
    }
    if (!track) {
        ErrorL << "FAIL: video track never became ready";
        return 1;
    }
    if (gop.empty()) {
        ErrorL << "FAIL: GOP cache empty - the snapshot path requires ENABLE_RTPPROXY "
                  "with rtp_proxy.gop_cache > 0 (default 1)";
        return 1;
    }
    InfoL << "drained " << gop.size() << " video frame(s) from the GOP cache";

    // Decode the cached GOP and save a JPEG, exactly as makeSnapInternal does.
    string err;
    if (!decodeGopToJpeg(track, gop, save_path, err)) {
        ErrorL << "FAIL: no decodable frame produced from GOP cache" << (err.empty() ? "" : (" - " + err));
        return 1;
    }

    auto size = File::fileSize(save_path.data());
    if (size == 0) {
        ErrorL << "FAIL: snapshot file is empty" << (err.empty() ? "" : (" - " + err));
        return 1;
    }
    if (!isJpeg(save_path)) {
        ErrorL << "FAIL: output is not a valid JPEG: " << save_path;
        return 1;
    }

    InfoL << "PASS: snapshot written to " << save_path << " (" << size << " bytes, valid JPEG)";
    reader->stopReadMP4();
    return 0;
}

#else // !(ENABLE_FFMPEG && ENABLE_MP4)

#include <iostream>

int main(int argc, char *argv[]) {
    std::cout << "test_getsnap skipped: build with -DENABLE_FFMPEG=ON -DENABLE_MP4=ON to run it" << std::endl;
    return 0;
}

#endif
